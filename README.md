# FSP Finder: AI-Powered Music Censoring Tool

FSP Finder (Foul Speech Pattern Finder) is a web application designed to automatically detect and censor explicit language in audio files. It leverages machine learning models for high-accuracy transcription and source separation, providing content creators and small broadcasters with a tool to quickly prepare audio for airplay.

Users can upload music tracks, review an AI-generated transcript with highlighted sections, and approve edits to generate a high-quality censored version where explicit vocals are muted while background instrumentals remain untouched.

## Key Features

* **Audio Source Separation:** Uses Demucs to separate vocals from instrumental tracks, ensuring only the vocal track is censored.
* **AI-Powered Transcription:** Employs a fine-tuned Whisper model to generate accurate, word-level timestamps for song lyrics.
* **Explicit Language Detection:** Analyzes transcripts to flag profanity, slurs, and other potentially offensive content.
* **Interactive Review:** Presents users with a full transcript where they can review and adjust suggested edits before finalizing.
* **High-Quality Censoring:** Creates the final track by muting the vocal stem at precise timestamps and recombining it with the original instrumental.
* **Scalable Task Queue:** Uses Celery and Redis to manage time-consuming audio processing tasks in the background, allowing multiple users to queue jobs simultaneously.

---

## Technology Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend API** | FastAPI | High-performance Python web framework for serving API endpoints. |
| **Task Queue** | Celery & Redis | Asynchronous task management for long-running analysis processes. |
| **ML: Transcription** | OpenAI Whisper | Automatic speech recognition with word-level timestamps. |
| **ML: Source Separation** | Demucs | Separates vocals from instrumentals in audio tracks. |
| **ML Framework** | PyTorch | Core machine learning library for running Whisper and Demucs. |
| **Deployment** | Docker & Docker Compose | Containerization for consistent development and production environments. |
| **Frontend** | HTML5 / Jinja2 / JavaScript | User interface for file upload, review, and download. |

---

## Getting Started

Follow these instructions to set up and run the application locally using Docker. This setup requires an NVIDIA GPU for the machine learning models.

### Prerequisites

* **Git:** To clone the repository.
* **NVIDIA GPU:** Required for efficient model inference.
* **NVIDIA Drivers:** Ensure you have the latest drivers installed on your host machine.
* **Docker Desktop:** To build and manage containers.
* **NVIDIA Container Toolkit:** To allow Docker containers to access the host machine's GPU.

### Installation and Launch

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create Environment File:**
    Create a file named `.env` in the root directory. This file holds API keys required by the application. Add the following keys (replace placeholders with real values):
    ```ini
    # For sending email notifications via contact form
    SENDGRID_API_KEY=YOUR_SENDGRID_API_KEY

    # For fetching official lyrics from Genius.com
    GENIUS_API_TOKEN=YOUR_GENIUS_API_TOKEN
    ```

3.  **Build and Run with Docker Compose:**
    This command builds the Docker image (pre-caching ML models in the process) and starts all services (web server, worker, and Redis database).

    ```bash
    docker-compose up --build
    ```
    * Use `docker-compose up -d --build` to run in detached mode (background).

---

## How to Use

1.  **Access the Application:** Open your web browser and navigate to `http://localhost`.
2.  **Upload Audio:** Drag and drop one or more audio files (`.mp3`, `.wav`, etc.) onto the upload form and click "Process audio."
3.  **Wait for Processing:** The app will show the job status. If the worker is busy, new jobs will display a "Queued" status before changing to "Processing."
4.  **Review Edits:** On the results page, review the generated transcript. Explicit words marked for censoring will be highlighted. You can toggle individual words or lines on/off.
5.  **Download Final Track:** Click the download button for each track to receive the final censored `.mp3` file.

## Project Structure Overview
```text
.
├── docker-compose.yml     # Orchestrates all services (web, worker, redis)
├── Dockerfile             # Blueprint for building the application container
├── requirements.txt       # Python dependencies
├── download_models.py     # Script to pre-cache ML models during build time
├── app.py                 # FastAPI application: routes, API endpoints, user interaction logic
├── celery_worker.py       # Celery configuration and definition of the main analysis task
├── ml_logic.py            # Core machine learning logic for Demucs, Whisper, and text processing
├── static/                # CSS stylesheets and images
└── templates/             # HTML templates (index.html, results.html, etc.)
```
