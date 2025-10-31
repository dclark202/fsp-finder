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
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import OutputFixingParser

SHARED_ARTIFACTS_PATH = "/job_artifacts"

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
LLM_MODEL_ID = "/model-cache/gemma-2-9b-it"
WHISPER_BASE_MODEL = "openai/whisper-medium.en"
WHISPER_FT_MODEL_PATH = '/model-cache/whisper-medium-ft'
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
def combine_audio(path1, path2, outpath, format='mp3'):
    audio1 = AudioSegment.from_file(path1, format='wav')
    audio2 = AudioSegment.from_file(path2, format='wav')
    combined_audio = audio1.overlay(audio2)

    if format == 'mp3': combined_audio.export(outpath, bitrate="256k", format=format)
    else: combined_audio.export(outpath, format=format)


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



def analyze_audio(audio_path, model, device, use_vad=False):
    """
    Separates the vocals stem and evaluates it via Whisper
    """
    run_id = str(uuid.uuid4())
    run_temp_dir = os.path.join(SHARED_ARTIFACTS_PATH, run_id)
    os.makedirs(run_temp_dir, exist_ok=True)
    
    source_path = os.path.abspath(audio_path)
    temp_audio_path = os.path.join(run_temp_dir, 'temp_audio.mp3')
    shutil.copy(source_path, temp_audio_path)

    metadata = get_metadata(temp_audio_path)
    metadata['genius_url'] = get_genius_url(metadata['artist'], metadata['title'])
    metadata['genius_lyrics'] = get_genius_lyrics(metadata['artist'], metadata['title'])

    print()
    demucs.separate.main([
        "--two-stems", "vocals", # separate vocals from non-vocals
        "--shifts", "2", # Two shifts for better quality vocals separation
        "-j", "8", # num parallel jobs (on CPU?)
        "-n", "mdx_extra", # mdx_extra transcription model
        "-o", run_temp_dir, # set the output directory
        temp_audio_path # input file
    ])
    
    demucs_out_name = os.path.splitext(os.path.basename(temp_audio_path))[0]
    vocals_path = os.path.join(run_temp_dir, "mdx_extra", demucs_out_name, "vocals.wav")
    no_vocals_path = os.path.join(run_temp_dir, "mdx_extra", demucs_out_name, "no_vocals.wav")

    audio = whisper.load_audio(vocals_path) # change to preprocessed_path to use exp settings

    print(f'\nTranscribing audio...')
    transcribe_options = {
        'beam_size': 5, 
        'best_of': 5, 
        'remove_empty_words': True,
        'temperature': (0.0, 0.2, 0.4, 0.6, 0.8), 
        'language': "en", 
        'task': 'transcribe'
    }

    if use_vad: transcribe_options['vad'] = 'silero' # Cached from download_models.py
        
    result = whisper.transcribe(model, audio, **transcribe_options)
    
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


def backup_censoring(text_tokens, custom_profanity_list=""):
    """
    Searches for always explicit content and ensures that it is always flagged
    """
    backup_explicit_ids = set()
    prev_word = ''

    all_sing_curse_words = singular_curse_words.copy()

    if custom_profanity_list:
        custom_words = {word.strip().lower() for word in custom_profanity_list.split(',')}
        all_sing_curse_words.update(custom_words)

    for j, word in enumerate(text_tokens):
        cleaned_word = remove_punctuation(word)
        
        # Always explicit content identified by substrings
        is_explicit = any(curse in cleaned_word for curse in default_curse_words)

        if cleaned_word in all_sing_curse_words: 
            # If a word matches **exactly** one of the added words or the few short words identified in sing_curse_words
            backup_explicit_ids.add(j)

        # Handle two word clusters ex: "god dam*", "mother fuck*"
        elif ('dam' in cleaned_word and prev_word == 'god') or ('fuck' in cleaned_word and prev_word == 'mother') or (cleaned_word == 'off' and prev_word == 'jerk'):
            backup_explicit_ids = backup_explicit_ids | {j-1, j}

        # To not double count, perform this check last
        elif is_explicit: backup_explicit_ids.add(j)

        # For two word phrases
        prev_word = cleaned_word
        
    return backup_explicit_ids


def process_transcription(transcription_result: dict) -> List[dict]:
    """
    Converts the raw Whisper transcription result into a structured list of segments.
    This function no longer applies the censoring, that has been moved to a new function
    """
    full_transcript = []
    raw_transcript = transcription_result.get("segments", [])
    
    i = 0
    for segment in raw_transcript:
        segment_words = []
        j = 0
        for word_info in segment.get('words', []):
            word_text = word_info.get('text', '').strip()
            if not word_text: continue
            
            start_time, end_time = float(word_info['start']), float(word_info['end'])

            if end_time - start_time < .1: continue 

            word_id = (i,j)
            word_data = {'id': word_id, 'text': word_text, 'start': start_time, 'end': end_time}
            segment_words.append(word_data)
            j += 1

        if not segment_words: continue

        i += 1
        line_text = ' '.join([d['text'] for d in segment_words])
        full_transcript.append({
            'line_words': segment_words, 
            'line_text': line_text, 
            'start': segment['start'], 
            'end': segment['end']
        })
            
    return full_transcript


def apply_censoring_logic(transcript: List[dict], llm_chain=None, profanity_list: str = "") -> List[tuple]:
    """
    Applies censoring logic to a complete transcript timeline and returns a list of IDs to mute.
    """
    ids_to_mute = []
    speech_segments = [seg for seg in transcript if seg.get('type') in ['speech', 'speech_recovered']]

    if llm_chain:
        lines_to_analyze = [{"text_to_analyze": line['line_text']} for line in speech_segments]

        try:
            print('\nCalling the LLM for explicit content detection... (this takes awhile)')
            llm_outputs = llm_chain.batch(lines_to_analyze, config={"max_concurrency": 8}) 

        except Exception as e:
            print(f"The LLM failed spectacularly")
            llm_outputs = [None] * len(lines_to_analyze)
        
        speech_segments = [line for line in transcript if line.get('type') in ['speech', 'speech_recovered']]

        for i, llm_output in enumerate(llm_outputs):
            line = speech_segments[i]
            new_ids = []
            text_tokens = [d['text'].strip().lower() for d in line['line_words']]
            
            explicit_ids = backup_censoring(text_tokens, profanity_list)

            # Ensure llm_output conforms to proper formatting
            if not isinstance(llm_output, Exception):
                explicit_phrases = llm_output.get('explicit_phrases_found', [])
                for phrase in explicit_phrases:
                    try:
                        phrase_tokens = phrase.split()
                        n = len(phrase_tokens)
                        for j in range(len(text_tokens) - n + 1):
                            if [token.lower() for token in text_tokens[j:j+n]] == [p_token.lower() for p_token in phrase_tokens]:
                                explicit_ids.update(range(j, j+n))
                        
                    except: continue
            
            # Improved logic for identifying the ids for the flagged words
            for word_index in explicit_ids:
                if word_index < len(line['line_words']):
                    original_word_id = line['line_words'][word_index]['id']
                    new_ids.append(original_word_id)

                ids_to_mute.extend(new_ids)
    
    ## If not using the LLM, just do the backup censoring
    else:                  
        for line in speech_segments:
            new_ids = []
            text_tokens = [d['text'].strip().lower() for d in line['line_words']]
            explicit_ids = backup_censoring(text_tokens, profanity_list)

            for word_index in explicit_ids:
                if word_index < len(line['line_words']):
                    original_word_id = line['line_words'][word_index]['id']
                    new_ids.append(original_word_id)

                ids_to_mute.extend(new_ids)

    return ids_to_mute


def apply_censoring(analysis_state, ids_to_censor, format='mp3'):
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
    output_path = os.path.join(analysis_state['temp_dir'], f"{base_name}-edited.{format}")

    combine_audio(silenced_vocals_path, analysis_state['no_vocals_path'], output_path, format=format)
    transfer_metadata(analysis_state['original_audio_path_copy'], output_path)

    return output_path


def process_untranscribed_gaps(audio_path: str, transcription_result: dict, whisper_model, device: str, silence_threshold_dbfs: float = -30.0, min_gap_seconds: float = 5.0) -> List[dict]:
    """
    Finds untranscribed gaps, re-runs them through Whisper, and returns both
    recovered speech and any remaining untranscribed (but not silent) gaps.
    """
    final_gap_segments = []

    try: audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio file for gap recovery: {e}")
        return final_gap_segments

    initial_segments = transcription_result.get("segments", [])
    if not initial_segments: return final_gap_segments

    # Helper function to process a potential gap
    def re_process_gap(start_ms, end_ms):
        this_gap_start = start_ms/1000
        this_gap_end = end_ms / 1000.0

        # Skip sections that are short or silent
        gap_duration_s = (end_ms - start_ms) / 1000.0
        if gap_duration_s < min_gap_seconds: return

        gap_audio = audio[start_ms:end_ms]
        if gap_audio.dBFS < silence_threshold_dbfs: return 

        print(f"--> Found untranscribed sound from {start_ms/1000:.2f}s to {end_ms/1000:.2f}s. Re-processing...")

        # Format audio for processing
        gap_audio = gap_audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(gap_audio.get_array_of_samples()).astype(np.float32) / 32768.0

        re_transcribe_options = {
            'beam_size': 8, # Higher beam_size and best_of for better transcription
            'best_of': 8, 
            'remove_empty_words': True,
            'language': "en", 
            'task': 'transcribe', 
            'temperature': (0.0, 0.2, 0.4, 0.6),
        }  
        
        re_result = whisper.transcribe(whisper_model, samples, **re_transcribe_options)

        if not re_result or not re_result.get("segments"):
            # print("--> Second pass yielded no new transcription. Flagging as a warning.")
            
            if gap_duration_s >= min_gap_seconds:
                final_gap_segments.append({
                    "start": start_ms / 1000.0, 
                    "end": end_ms / 1000.0, 
                    "type": "untranscribed_final"
                })

            return

        recovered_segments = re_result["segments"]
        # print(f"--> Recovered {len(recovered_segments)} new segment(s) in the gap.")
        
        last_rec_end_s = this_gap_start

        # This is a little hacky but ensures that we get a unique id for each of the segments
        i = -start_ms

        for segment in recovered_segments:
            rec_start_s = this_gap_start + segment['start']
            
            if rec_start_s - last_rec_end_s >= min_gap_seconds:
                final_gap_segments.append({"start": last_rec_end_s, "end": rec_start_s, "type": "untranscribed_final"})

            segment_words = []

            j = 0
            for word_info in segment.get('words', []):
                word_text = word_info.get('text', '').strip()
                if not word_text: continue
                
                start_time, end_time = float(word_info['start']), float(word_info['end'])

                if end_time - start_time < .1: continue 

                word_id = (i, j)
                word_data = {
                    'id': word_id, 
                    'text': word_text, 
                    'start': start_time + this_gap_start, 
                    'end': end_time + this_gap_start
                }

                segment_words.append(word_data)
                j += 1

            if not segment_words: continue
            i -= 1

            line_text = ' '.join([d['text'] for d in segment_words])

            final_gap_segments.append({
                'line_words': segment_words, 
                'line_text': line_text, 
                'start': segment['start'] + this_gap_start, 
                'end': segment['end'] + this_gap_start,
                'type': 'speech_recovered'
            })
            last_rec_end_s = this_gap_start + segment['end']

        # Identify any further remaining gaps that have gone untranscribed and flag them
        if this_gap_end - last_rec_end_s >= min_gap_seconds:
            final_gap_segments.append({"start": last_rec_end_s, "end": this_gap_end, "type": "untranscribed_final"})

    # Initialize the gap times
    last_end_ms = 0

    # Reprocess the audio at the identified gaps
    for segment in initial_segments:
        start_ms = int(segment['start'] * 1000)

        if start_ms > last_end_ms: 
            re_process_gap(last_end_ms, start_ms)

        last_end_ms = int(segment['end'] * 1000)
    
    audio_duration_ms = len(audio)
    if audio_duration_ms > last_end_ms: re_process_gap(last_end_ms, audio_duration_ms)
    
    return final_gap_segments


def create_llm_chain():
    """
    Loads the Gemma LLM with 4-bit quantization and builds the LangChain chain.
    This function is computationally expensive and should only be called when needed.
    """
    class ExplicitPhrase(BaseModel):
        explicit_phrases_found: List[str] = Field(description="A list of phrases from the text that are explicit, suggestive, or profane.")

    print(f"Loading LLM: {LLM_MODEL_ID}...")
    
    # This is saved as part of the model with download_models.py
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    LLM_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=os.getenv("HF_TOKEN"))
    LLM_MODEL = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        #quantization_config=quantization_config,
        device_map="auto",
        #token=os.getenv("HF_TOKEN")
    )

    gemma_template = ("{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}{% elif message['role'] == 'model' %}{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}")
    LLM_TOKENIZER.chat_template = gemma_template

    # Create HF pipeline with custom tokenizer (necessary)
    pipe = pipeline(
        "text-generation",
        model=LLM_MODEL,
        tokenizer=LLM_TOKENIZER,
        max_new_tokens=512,
        device=None 
    )
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)

    # This has been the best performing prompt
    prompt_str = """
    You are an AI content moderator for a US radio station. Your goal is to identify words and phrases that would be considered 'indecent' or 'profane' under FCC broadcast standards, making them unsuitable for airplay.

    You must consider the context. Do not flag words that are merely references to mature themes (like drugs or weapons) unless they are used in a particularly graphic, gratuitous, or shocking manner. Differentiate between a mere mention and content designed to shock.

    **Text to Analyze:**
    "{text_to_analyze}"

    Return a single JSON object with one key: "explicit_phrases_found" containing a list of each word or phrase EXACTLY as it appears in the text.

    Provide only the raw JSON object as your final response.
    """

    chat_prompt = ChatPromptTemplate.from_messages([("user", prompt_str)])

    json_parser = JsonOutputParser(pydantic_object=ExplicitPhrase)
    fixing_parser = OutputFixingParser.from_llm(parser=json_parser, llm=hf_pipeline) # Fixes issues with the JSON parsing

    # Read from left to right (weird)
    chain = chat_prompt | hf_pipeline | fixing_parser

    return chain