# FSP Finder: AI-Powered Music Censoring Tool

FSP Finder (Foul Speech Pattern Finder) is a web application designed to automatically detect and censor explicit language in audio files. It leverages machine learning models for high-accuracy transcription and source separation, providing content creators and small broadcasters with a tool to quickly prepare audio for airplay.

Users can upload music tracks, review an AI-generated transcript with highlighted sections, and approve edits to generate a high-quality censored version where explicit vocals are muted while background instrumentals remain untouched.

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
    git clone github.com/dclark202/fsp-finder.git
    cd fsp-finder
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
