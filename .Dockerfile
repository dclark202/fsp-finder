# --- Stage 1: Define the Base Image ---
# Use an official NVIDIA CUDA runtime image instead of python:slim.
FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04

# --- Stage 2: Install Python and System Dependencies ---
# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    # ffmpeg for mp3 processing
    ffmpeg \ 
    build-essential && \
    # Add deadsnakes PPA to get access to newer Python versions
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    # Install Python 3.11 runtime and virtual environment module
    apt-get install -y python3.11 python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

# Install pip for Python 3.11 using ensurepip, then upgrade it
RUN python3.11 -m ensurepip && \
    python3.11 -m pip install --upgrade pip setuptools wheel

# --- Stage 3: Install Python Application Dependencies ---
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install torch with GPU support 
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126

# Install Python packages using requirements.txt
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# --- Stage 4: Copy Application Code & Cache Models ---
COPY . .
RUN python3.11 download_models.py

# Create application-specific directories before changing permissions
RUN mkdir -p /app/uploads \
    && mkdir -p /app/training_data \
    && mkdir -p /model-cache \
    && mkdir -p /job_artifacts

# Change ownership of the /app directory and cache directory to the new non-root user
RUN chown -R appuser:appgroup /app \
    && chown -R appuser:appgroup /model-cache \
    && chown -R appuser:appgroup /job_artifacts

# Switch to the non-root user for all subsequent commands
USER appuser

# Expose port for FastAPI application
EXPOSE 8000