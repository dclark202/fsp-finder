import os
import json
import shutil
import traceback
from typing import List, Optional

# API
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query, Form, Path
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import redis

from pydantic import BaseModel

# Celery for background processing
from celery.result import AsyncResult
from ml_logic import apply_censoring  
from celery_worker import celery_app, UPLOAD_FOLDER

# Email logic
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, ReplyTo

app = FastAPI(title="FSP Finder Backend")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

redis_client = redis.Redis(host='redis', port=6379, db=1, decode_responses=True)
REDIS_EXPIRATION_SECONDS = 900

class FinalizeRequest(BaseModel):
    job_id: str
    ids_to_censor: List[List[int]] = None 

@app.api_route("/", methods=["GET", "HEAD"])
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/contact")
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.get("/terms")
async def terms(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})

@app.get("/methodology")
async def methodology(request: Request):
    return templates.TemplateResponse("methodology.html", {"request": request})

# --- Contact Form Logic ---

@app.post("/contact")
async def handle_contact_form(name: str = Form(...), email: str = Form(...), message: str = Form(...)):
    try:
        email_message = Mail(
            from_email='contact@fspfinder.com',
            to_emails='contact@fspfinder.com',
            subject=f'FSP Finder - {name}',
            html_content=f'<strong>Name:</strong> {name}<br>'
                         f'<strong>Email:</strong> {email}<br><br>'
                         f'<strong>Message:</strong><p>{message}</p>')
        email_message.reply_to = ReplyTo(email, name)
        
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        sg.send(email_message)
        
        return JSONResponse({'status': 'success', 'message': 'Thank you for your message!'})
    
    except Exception as e:
        print("--- AN ERROR OCCURRED IN THE /contact ENDPOINT ---")
        traceback.print_exc() # This will print the full, detailed error traceback
        return JSONResponse({'status': 'error', 'message': 'An error occurred.'}, status_code=500)

# --- Core Application API Endpoints ---

@app.post("/analyze")
async def analyze_files(files: List[UploadFile] = File(...), profanity_list: str = Form(""), use_vad: bool = Form(False), llm_detection: bool = Form(False)):
    # Import the task function here to avoid circular import issues at startup
    from celery_worker import analysis_task
    
    job_ids = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        task = analysis_task.delay(
            file_path, 
            profanity_list=profanity_list, 
            use_vad=use_vad, 
            llm_detection=llm_detection
        )

        job_ids.append(task.id)
        
        redis_client.setex(f"status_{task.id}", 3600, "queued")
        
    return JSONResponse({"job_ids": job_ids})


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    task_result = AsyncResult(job_id, app=celery_app)

    if task_result.ready(): # Task is done
        if task_result.successful():
            return {"status": "complete"}
        else:
            return {"status": "failed", "error": str(task_result.info)}

    # Check our custom Redis status
    custom_status = redis_client.get(f"status_{job_id}")
    if custom_status == "processing":
        return {"status": "processing"}
    else:
        # If the task isn't ready and not marked as 'processing', it must be in the queue.
        return {"status": "queued"}


@app.get("/results")
async def get_results_page(request: Request, job_ids: str = Query(...)):
    job_id_list = job_ids.split(',')
    results_list = []
    
    for job_id in job_id_list:
        task_result = AsyncResult(job_id, app=celery_app)
        if task_result.successful():
            result_data = task_result.get()
            result_data['filename'] = result_data.get('original_filename', 'Unknown')
            result_data['job_id'] = job_id
            results_list.append(result_data)
            redis_client.setex(job_id, 3600, json.dumps(result_data))
        else:
            # Add a placeholder for any failed jobs
            results_list.append({
                "job_id": job_id, "filename": f"Job {job_id[:8]}...",
                "metadata": {"artist": "Error", "title": "Processing Failed"},
                "transcript": [{"line_text": "Could not retrieve result."}]
            })

    return templates.TemplateResponse("results.html", {"request": request, "results_list": results_list})


@app.post("/keep-alive/{job_id}")
async def keep_alive(job_id: str = Path(...)):
    """
    Extends the expiration time for a job's data in Redis.
    """
    # Check for the main job data key and the status key
    if redis_client.exists(job_id):
        redis_client.expire(job_id, REDIS_EXPIRATION_SECONDS)

    if redis_client.exists(f"status_{job_id}"):
        redis_client.expire(f"status_{job_id}", REDIS_EXPIRATION_SECONDS)

    return JSONResponse({"status": "ok", "message": f"Session for job {job_id} extended."})


@app.post("/finalize")
async def finalize_file(job_id: str = Form(...), ids_to_censor: str = Form(...)):
    cached_result_json = redis_client.get(job_id)

    if not cached_result_json:
        raise HTTPException(status_code=404, detail="Job data not found. It may have expired.")

    analysis_state = json.loads(cached_result_json)
    
    # Get the paths for cleanup before the main logic
    # temp_dir = analysis_state.get('temp_dir')
    # initial_upload_path = os.path.join(UPLOAD_FOLDER, analysis_state.get('original_filename', ''))
    
    # The list of IDs comes from the form as a JSON string, so we parse it
    final_ids_to_censor = json.loads(ids_to_censor)
    
    try:
        data_to_save = {
            "metadata": analysis_state.get('metadata'),
            "transcript": analysis_state.get('transcript'),
            "final_censored_ids": final_ids_to_censor
        }
        
        save_dir = "training_data"
        os.makedirs(save_dir, exist_ok=True)
        
        base_name = os.path.splitext(analysis_state.get('original_filename', 'file'))[0]
        json_filename = f"{base_name}_{job_id[:8]}.json"
        save_path = os.path.join(save_dir, json_filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        print(f"Saved training data to {save_path}")
        
    except Exception as e:
         print(f"ERROR: Could not save training data for job {job_id}. Reason: {e}")

    output_path = apply_censoring(analysis_state, final_ids_to_censor)
    
    if not output_path:
        raise HTTPException(status_code=400, detail="No content was marked for censoring.")
    
    # tasks = BackgroundTasks()

    # Return the file, passing the cleanup task to be run AFTER the response is sent
    return FileResponse(
        path=output_path, 
        media_type='audio/mpeg', 
        filename=os.path.basename(output_path),
        # background=tasks  
    )
