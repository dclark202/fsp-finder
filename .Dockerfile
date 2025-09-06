# --- Stage 1: Define the Base Image ---
# Use an official NVIDIA CUDA runtime image instead of python:slim.
# This example uses CUDA 12.1. Adjust the tag based on your GPU driver compatibility.
# Format: nvidia/cuda:<cuda_version>-cudnn<cudnn_version>-runtime-<os_version>
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# --- Stage 2: Install Python and System Dependencies ---
# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-setuptools \
    git \
    && rm -rf /var/lib/apt/lists/*

# Link python3.10 to python3 and python, and pip3.10 to pip3 and pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# --- Stage 3: Install Python Application Dependencies ---
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install Python packages using requirements.txt (which now includes the GPU-specific torch command)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 4: Copy Application Code & Cache Models ---
COPY . .
RUN python download_models.py

# Expose port for FastAPI application
EXPOSE 8000