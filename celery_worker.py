# celery_worker.py
import os
import shutil
from celery import Celery
from ml_logic import analyze_audio, process_transcription, calculate_wer
import torch
import whisper_timestamped as whisper
import redis

WHISPER_MODEL = None

# Define a shared upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Celery configuration
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0' 
)
redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)


@celery_app.task(bind=True)
def analysis_task(self, file_path: str):
    """
    Runs the full analysis and returns the results dictionary.
    This runs in a separate Celery worker process.
    """
    job_id = self.request.id
    redis_client.setex(f"status_{job_id}", 3600, "processing")

    global WHISPER_MODEL 
    analysis_state = {}
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if WHISPER_MODEL is None:
        print("CELERY WORKER: Loading Whisper model for the first time in this process...")
        WHISPER_MODEL = whisper.load_model('whisper-medium-ft', device=DEVICE)
        print("CELERY WORKER: Whisper model loaded successfully.")

    analysis_state = {}

    analysis_state = analyze_audio(file_path, WHISPER_MODEL, DEVICE)
    processed_data = process_transcription(analysis_state['transcription_result'], llm_chain=None)
    analysis_state.update(processed_data)

    transcript_text = " ".join([word['text'] for seg in analysis_state['transcript'] for word in seg['line_words']])
    analysis_state['metadata']['wer_score'] = calculate_wer(analysis_state['metadata']['genius_lyrics'], transcript_text)

    del analysis_state['transcription_result'] 
    
    print(f"CELERY TASK: Analysis complete for {file_path}")
    return analysis_state 