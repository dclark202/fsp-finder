# celery_worker.py
import os
import shutil
import time
from celery import Celery
from ml_logic import analyze_audio, process_transcription, calculate_wer, create_llm_chain, apply_censoring_logic, process_untranscribed_gaps
import torch
import whisper_timestamped as whisper
import redis
from celery.schedules import crontab

from ml_logic import WHISPER_FT_MODEL_PATH

WHISPER_MODEL = None
LLM_CHAIN = None

SHARED_ARTIFACTS_PATH = "/job_artifacts"
EXPIRATION_HOURS = 2 # Files older than 2 hours will be deleted
EXPIRATION_SECONDS = EXPIRATION_HOURS * 3600

REDIS_SESSION_SECONDS = 900

# Define a shared upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Celery configuration
celery_app = Celery(
    'tasks',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/0' 
)

celery_app.conf.beat_schedule = {
    'cleanup-stale-files-hourly': {
        'task': 'celery_worker.cleanup_stale_files',
        'schedule': crontab(minute=0, hour='*/1'), # Run at minute 0 of every hour
    },
}

# Same Redis client and the app.py
redis_client = redis.Redis(host='redis', port=6379, db=1, decode_responses=True)

@celery_app.task(bind=True)
def analysis_task(self, file_path: str, profanity_list: str, use_vad: bool, llm_detection: bool):
    """
    Runs the full analysis and returns the results dictionary.
    This runs in a separate Celery worker process.
    """
    job_id = self.request.id
    redis_client.setex(f"status_{job_id}", REDIS_SESSION_SECONDS, "processing")

    global WHISPER_MODEL, LLM_CHAIN
    analysis_state = {}
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if WHISPER_MODEL is None:
        print("CELERY WORKER: Loading Whisper model for the first time in this process...")
        WHISPER_MODEL = whisper.load_model(WHISPER_FT_MODEL_PATH, device=DEVICE)
        print("CELERY WORKER: Whisper model loaded successfully.")

    llm_chain_to_pass = None
    if llm_detection:
        if LLM_CHAIN is None:
            print("CELERY WORKER: LLM detection enabled. Loading LLM for the first time in this process...")
            LLM_CHAIN = create_llm_chain()
            print("CELERY WORKER: LLM loaded successfully.")
        llm_chain_to_pass = LLM_CHAIN

    analysis_state = analyze_audio(file_path, WHISPER_MODEL, DEVICE, use_vad=use_vad)
    initial_transcript = process_transcription(analysis_state['transcription_result'])

    # Add a 'type' to existing speech segments
    for segment in initial_transcript:
        segment['type'] = 'speech'

    print('Checking for unprocessed vocals ...')

    final_gap_segments = process_untranscribed_gaps(
        audio_path=analysis_state['vocals_path'],
        transcription_result=analysis_state['transcription_result'],
        whisper_model=WHISPER_MODEL,
        device=DEVICE
    )

    # Combine the two lists and sort by start time to create a complete timeline
    full_timeline = sorted(
        initial_transcript + final_gap_segments,
        key=lambda x: x['start']
    )
    
    # for item in full_timeline:
    #     print(item)

    ids_to_mute = apply_censoring_logic(
        transcript=full_timeline,
        llm_chain=llm_chain_to_pass, 
        profanity_list=profanity_list
    )

    # Add the ids to mute
    analysis_state['initial_explicit_ids'] = ids_to_mute

    # Add the full transcript
    analysis_state['transcript'] = full_timeline

    transcript_text = " ".join([seg['line_text'] for seg in analysis_state['transcript'] if seg.get('type') in ['speech', 'speech_recovered']])
    analysis_state['metadata']['wer_score'] = calculate_wer(analysis_state['metadata']['genius_lyrics'], transcript_text)

    # Clean up the large, raw result before sending it back
    del analysis_state['transcription_result']
    
    print(f"CELERY TASK: Analysis complete for {file_path}")
    return analysis_state


@celery_app.task
def cleanup_stale_files():
    """Scans artifact and upload directories and removes files older than EXPIRATION_HOURS."""
    print(f"--- Running scheduled cleanup: Deleting files older than {EXPIRATION_HOURS} hours ---")
    now = time.time()

    # Clean up job artifact directories (/job_artifacts)
    if os.path.exists(SHARED_ARTIFACTS_PATH):
        for dirname in os.listdir(SHARED_ARTIFACTS_PATH):
            dirpath = os.path.join(SHARED_ARTIFACTS_PATH, dirname)
            if os.path.isdir(dirpath):
                try:
                    # Get last modification time of the directory itself
                    modified_time = os.path.getmtime(dirpath)
                    if (now - modified_time) > EXPIRATION_SECONDS:
                        print(f"Cleaning up stale artifact directory: {dirpath}")
                        shutil.rmtree(dirpath)
                except FileNotFoundError:
                    continue # File might have been deleted by a parallel process

    # Clean up original uploads (/app/uploads)
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                try:
                    modified_time = os.path.getmtime(filepath)
                    if (now - modified_time) > EXPIRATION_SECONDS:
                        print(f"Cleaning up stale upload file: {filepath}")
                        os.remove(filepath)
                except FileNotFoundError:
                    continue
    
    return f"Cleanup complete. Removed items older than {EXPIRATION_HOURS} hours."