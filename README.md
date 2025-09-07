# FSP Finder: AI-Powered Music Censoring Tool

FSP Finder (Foul Speech Pattern Finder) is a web application designed to automatically detect and censor explicit language in audio files. It leverages machine learning models for high-accuracy transcription and source separation, providing content creators and small broadcasters with a tool to quickly prepare audio for airplay.

Users can upload music tracks, review an AI-generated transcript with highlighted sections, and approve edits to generate a high-quality censored version where explicit vocals are muted while background instrumentals remain untouched.

---

## Getting Started

Follow these instructions to set up and run the application locally using Docker. This setup requires an NVIDIA GPU for the machine learning models.

### Prerequisites

* **CUDA capable GPU:** Required for efficient model inference (>12 GB VRAM recommended)
* **Docker Desktop:** To build and manage containers (if using Windows, you will also need Windows Subsystem for Linux)

### Installation and Launch

1.  **Clone the Repository:**
    ```bash
    git clone github.com/dclark202/fsp-finder.git
    cd fsp-finder
    ```

2.  **Create Environment File:**
    Create a file named `.env` in the root directory. This file holds API keys required by the application (Note: the link on the Contact us page will not work if using this app locally).
    ```ini
    # (Optional) For fetching official lyrics from Genius.com
    GENIUS_API_TOKEN=YOUR_GENIUS_API_TOKEN

    # (Optional) For using the LLM to detect edge case explicit content
    # Requires access to Google Gemma 2
    HUGGING_FACE_HUB_TOKEN=YOUR_HF_TOKEN
    ```

3.  **Build and Run with Docker Compose:**
    This command builds the Docker image (pre-caching ML models in the process) and starts all services (web server, worker, and Redis database).

    ```bash
    docker-compose up --build
    ```
    * Use `docker-compose up -d --build` to run in detached mode (background).
