import os, re, json, shutil, tempfile, uuid
import torch
from transformers import WhisperForConditionalGeneration
from peft import PeftModel
import whisper_timestamped as whisper
import demucs.separate
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3
import lyricsgenius
import jiwer

# IMPORTS FOR USING LLM
# import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# import numpy as np
# from pydantic import BaseModel, Field, ValidationError
# from typing import List, Optional
# from langchain_huggingface import HuggingFacePipeline
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.runnables import RunnableLambda
# from langchain.output_parsers import OutputFixingParser


# For censoring 
default_curse_words = {
    'fuck', 'shit', 'piss', 'bitch', 'nigg', 'dyke', 'cock', 'faggot', 
    'cunt', 'tits', 'pussy', 'dick', 'asshole', 'whore', 'goddam',
    'douche', 'chink', 'tranny', 'jizz', 'kike', 'gook', 'cocksucker'
}

singular_curse_words = {
    'fag', 'cum', 'clit', 'wank', 'ho', 'hoes'
}

# --- Model & Device Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LLM_MODEL_ID = "google/gemma-2-9b-it"
WHISPER_BASE_MODEL = "openai/whisper-medium.en"
WHISPER_FT_MODEL_PATH = 'whisper-medium-ft'
LORA_CONFIG_PATH = './lora_config'


GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN")
if not GENIUS_API_TOKEN: print("WARNING: GENIUS_API_TOKEN environment variable not set. Genius API features will fail.")

genius = lyricsgenius.Genius(GENIUS_API_TOKEN, verbose=False, remove_section_headers=True) if GENIUS_API_TOKEN else None


###############################################################################################
### CORE LOGIC & HELPER FUNCTIONS
###############################################################################################

def load_whisper_model(model_path, lora_config, base_model_name):
    """Creates the full fine-tuned Whisper model from LoRA weights if it doesn't exist."""
    if os.path.exists(f'./{model_path}/model.safetensors'):
        print(f'Fine-tuned model at {model_path} already exists.')
        return
    
    print(f'Fine-tuned model not found. Creating model from LoRA configuration at {lora_config}')
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, lora_config)
    model = model.merge_and_unload()
    model.save_pretrained(model_path, save_serialization=False)
    print(f'Whisper model from {lora_config} saved at {model_path}')

# Removes all punctuation and returns lower case only words
def remove_punctuation(s):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    return s.lower()

# For silencing the audio tracks at the indicated times
def silence_audio_segment(input_audio_path, output_audio_path, times):
    audio = AudioSegment.from_file(input_audio_path)
    for (start_ms, end_ms) in times:
        before_segment = audio[:start_ms]
        target_segment = audio[start_ms:end_ms] - 60
        after_segment = audio[end_ms:]
        audio = before_segment + target_segment + after_segment
    audio.export(output_audio_path, format='wav')

# For combining the vocals and instrument stems once the censoring has been applied
def combine_audio(path1, path2, outpath):
    audio1 = AudioSegment.from_file(path1, format='wav')
    audio2 = AudioSegment.from_file(path2, format='wav')
    combined_audio = audio1.overlay(audio2)
    combined_audio.export(outpath, format="mp3")

# Extracts metadata from the original song
def get_metadata(original_audio_path):
    try:
        audio_orig = EasyID3(original_audio_path)
        metadata = {'title': audio_orig.get('title', [None])[0], 'artist': audio_orig.get('artist', [None])[0], 'album': audio_orig.get('album', [None])[0], 'year': audio_orig.get('date', [None])[0]}
    except Exception:
        metadata = {'title': 'N/A', 'artist': 'N/A', 'album': 'N/A', 'year': 'N/A'}
    return metadata

# Transfers metadata between two songs
def transfer_metadata(original_audio_path, edited_audio_path):
    try:
        audio_orig = EasyID3(original_audio_path)
        audio_edit = EasyID3(edited_audio_path)
        for key in audio_orig.keys():
            audio_edit[key] = audio_orig[key]
        audio_edit.save()

    except Exception as e:
        print(f"Could not transfer metadata: {e}")

# Lookup url on genius of lyrics for given song
def get_genius_url(artist, song_title):
    if not genius or not artist or not song_title or artist == 'N/A' or song_title == 'N/A': return None
    try:
        song = genius.search_song(song_title, artist)
        return song.url if song else None
    except Exception: return None

# It's called calculate_wer but I'm actually using *mer*
def calculate_wer(ground_truth, hypothesis):
    if not ground_truth or not hypothesis or "not available" in ground_truth.lower(): return None
    
    try:
        transformation = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ExpandCommonEnglishContractions(), jiwer.RemoveEmptyStrings()])
        error = jiwer.mer(transformation(ground_truth), transformation(hypothesis))
        return f"{error:.3f}"
    
    except Exception: return "Error"

# Gets the lyrics from genius for a given song
def get_genius_lyrics(artist, song_title):
    if not genius or not artist or not song_title or artist == 'N/A' or song_title == 'N/A': return "Lyrics not available (missing metadata or Genius API key)."
    
    try:
        song = genius.search_song(song_title, artist)
        return song.lyrics if song else "Could not find lyrics on Genius."
    
    except Exception: return "An error occurred while searching for lyrics."

# Separate track via demucs, evaluate vocals with Whisper
def analyze_audio(audio_path, model, device):
    run_temp_dir = tempfile.mkdtemp()
    source_path = os.path.abspath(audio_path)
    temp_audio_path = os.path.join(run_temp_dir, 'temp_audio.mp3')
    shutil.copy(source_path, temp_audio_path)

    metadata = get_metadata(temp_audio_path)
    metadata['genius_url'] = get_genius_url(metadata['artist'], metadata['title'])
    metadata['genius_lyrics'] = get_genius_lyrics(metadata['artist'], metadata['title'])

    print()
    demucs.separate.main([
        "--two-stems", "vocals", # separate vocals from non-vocals
        "--shifts", "2", # two shifts. For better quality use more!
        "-j", "8", # num parallel jobs
        "-n", "mdx_extra", # mdx_extra transcription model
        "-o", run_temp_dir, # set the output directory
        temp_audio_path # input file
    ])
    
    demucs_out_name = os.path.splitext(os.path.basename(temp_audio_path))[0]
    vocals_path = os.path.join(run_temp_dir, "mdx_extra", demucs_out_name, "vocals.wav")
    no_vocals_path = os.path.join(run_temp_dir, "mdx_extra", demucs_out_name, "no_vocals.wav")

    audio = whisper.load_audio(vocals_path) # change to preprocessed_path to use exp settings

    print(f'\nTranscribing audio...')
    result = whisper.transcribe(
        model, 
        audio, 
        beam_size=5, 
        best_of=5, 
        remove_empty_words=True, # revomves nonsense from the end of the transcript
        #refine_whisper_precision=.8
        #vad='silero', # Use silero to remove sections of silence, 
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8), 
        language="en", 
        task='transcribe'
    )
    
    if device == 'cuda': torch.cuda.empty_cache()
    
    return {
        "temp_dir": run_temp_dir,
        "vocals_path": vocals_path,
        "no_vocals_path": no_vocals_path,
        "original_audio_path_copy": temp_audio_path,
        "original_filename": os.path.basename(source_path),
        "transcription_result": result,
        "metadata": metadata
    }

def backup_censoring(text_tokens):
    backup_explicit_ids = set()
    prev_word = ''

    for j, word in enumerate(text_tokens):
        cleaned_word = remove_punctuation(word)
        is_explicit = any(curse in cleaned_word for curse in default_curse_words)

        # Short words that can be substrings of nonsensitive words
        if cleaned_word in singular_curse_words: backup_explicit_ids.add(j)

        # Handle two word cluster "god dam*", "mother fuck*"
        elif ('dam' in cleaned_word and prev_word == 'god') or ('fuck' in cleaned_word and prev_word == 'mother') or (cleaned_word == 'off' and prev_word == 'jerk'):
            backup_explicit_ids = backup_explicit_ids | {j-1, j}

        # The majority of censored words will come from here
        elif is_explicit: backup_explicit_ids.add(j)

        prev_word = cleaned_word
        
    return backup_explicit_ids

def process_transcription(transcription_result, llm_chain):
    full_transcript = []
    ids_to_mute = []
    low_quality = []
    raw_transcript = transcription_result.get("segments", [])
    
    ## Create transcript
    i = 0
    for segment in raw_transcript:
        segment_words = []
        j = 0
        for word_info in segment.get('words', []):
            word_text = word_info.get('text', '').strip()
            if not word_text: continue
            
            start_time, end_time = float(word_info['start']), float(word_info['end'])

            # Filter out hallucinations with very low word length (100ms)
            if end_time - start_time < .1: continue 

            word_id = (i,j)
            word_data = {'id': word_id, 'text': word_text, 'start': start_time, 'end': end_time}
            segment_words.append(word_data)
            j += 1

        if not segment_words: continue

        i += 1
        line_text = ' '.join([d['text'] for d in segment_words])
        full_transcript.append({'line_words': segment_words, 'line_text': line_text, 'start': segment['start'], 'end': segment['end']})

        if segment['avg_logprob'] < -1.0: low_quality.append(i+1)
            
    if low_quality: print(f'Low quality sections: {low_quality}')

    ## Use LLM for edge case detection (disabled at the moment)
    line_errs = []
    
    if llm_chain:
        lines_to_analyze = [{"text_to_analyze": line['line_text']} for line in full_transcript]

        try:
            print('\nCalling the LLM for explicit content detection...')
            llm_outputs = llm_chain.batch(lines_to_analyze, config={"max_concurrency": 8}) 

        except Exception as e:
            print(f"The LLM failed spectacularly")
            llm_outputs = [e] * len(lines_to_analyze)
        
        for i, llm_output in enumerate(llm_outputs):
            #print(llm_output)
            line_data = full_transcript[i]
            text_tokens = [d['text'].strip().lower() for d in line_data['line_words']]
            
            explicit_ids = backup_censoring(text_tokens)

            # If the JSON parsing for the LLM fails
            if isinstance(llm_output, Exception):
                print(f'Parsing error at line {i+1}')
                line_errs.append(i)
            
            else:
                explicit_phrases = llm_output.get('explicit_phrases_found', [])
                for phrase in explicit_phrases:
                    try:
                        phrase_tokens = phrase.split()
                        n = len(phrase_tokens)
                        for j in range(len(text_tokens) - n + 1):
                            if [token.lower() for token in text_tokens[j:j+n]] == [p_token.lower() for p_token in phrase_tokens]:
                                explicit_ids.update(range(j, j+n))
                        
                    except: continue

            ids_to_mute.extend([(i,j) for j in sorted(list(explicit_ids))])
    
    ## If not using the LLM, just do the backup censoring
    else:
        for i, line in enumerate(full_transcript):
            text_tokens = [d['text'].strip().lower() for d in line['line_words']]
            explicit_ids = backup_censoring(text_tokens)
            ids_to_mute.extend([(i,j) for j in sorted(list(explicit_ids))])

    #print(ids_to_mute)

    return {
        "transcript": full_transcript,
        "initial_explicit_ids": ids_to_mute,
        "line_errs": line_errs
    }

def apply_censoring(analysis_state, ids_to_censor):
    if not ids_to_censor: return None
    
    ids_set = {tuple(item) for item in ids_to_censor}
    times_to_censor = []
    transcript = analysis_state.get('transcript', [])

    for segment in transcript:
        for word in segment.get('line_words', []):
            if tuple(word.get('id')) in ids_set:
                times_to_censor.append({'start': word['start'], 'end': word['end']})

    times_in_ms = [(int(t['start']*1000), int(t['end']*1000)) for t in times_to_censor]
    silenced_vocals_path = os.path.join(analysis_state['temp_dir'], "vocals_silenced.wav")
    silence_audio_segment(analysis_state['vocals_path'], silenced_vocals_path, times_in_ms)
    
    base_name = os.path.splitext(analysis_state['original_filename'])[0]
    output_path = os.path.join(analysis_state['temp_dir'], f"{base_name}-edited.mp3")

    combine_audio(silenced_vocals_path, analysis_state['no_vocals_path'], output_path)
    transfer_metadata(analysis_state['original_audio_path_copy'], output_path)

    return output_path