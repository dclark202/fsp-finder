FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    ffmpeg \ 
    build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

# Install pip for Python 3.11 using ensurepip, then upgrade it
RUN python3.11 -m ensurepip && \
    python3.11 -m pip install --upgrade pip setuptools wheel

# Install Python Application Dependencies
WORKDIR /app
ENV TORCH_HOME=/model-cache/torch
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install torch with GPU support first
RUN python3.11 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126

# Install Python packages using requirements.txt (leverages layer caching)
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files for the download script to run
COPY download_models.py .
COPY ml_logic.py .
COPY lora_config/ ./lora_config/

# Declare the build-time argument for the token
ARG HF_TOKEN

# Run the script to download and cache all models
RUN python3.11 download_models.py

# --- Application Setup ---
# Now copy the rest of the application code
COPY . .

# Create application-specific directories
RUN mkdir -p /app/uploads \
    && mkdir -p /app/training_data \
    && mkdir -p /job_artifacts

# Change ownership of all relevant directories to the non-root user
RUN chown -R appuser:appgroup /app \
    && chown -R appuser:appgroup /model-cache \
    && chown -R appuser:appgroup /job_artifacts

# Switch to the non-root user for runtime
USER appuser

# Expose port for FastAPI application
EXPOSE 8000