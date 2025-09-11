import os
import torch
import demucs.pretrained
from transformers import WhisperForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ml_logic import WHISPER_BASE_MODEL, WHISPER_FT_MODEL_PATH, LORA_CONFIG_PATH

def cache_whisper_model():
    """
    Loads the base model from Hugging Face, applies LoRA weights,
    and saves the merged, fine-tuned model locally.
    This ensures the model directory 'whisper-medium-ft' exists for the worker.
    """
    print(f"--- Caching Whisper Model ---")
    
    # Check if the fine-tuned model already exists.
    # Adjust condition if your saved model file has a different name.
    if os.path.exists(os.path.join(WHISPER_FT_MODEL_PATH, 'pytorch_model.bin')) or \
       os.path.exists(os.path.join(WHISPER_FT_MODEL_PATH, 'model.safetensors')):
        print(f"Fine-tuned model already found at '{WHISPER_FT_MODEL_PATH}'. Skipping creation.")
        return

    print(f"Creating fine-tuned model from base '{WHISPER_BASE_MODEL}' and LoRA config '{LORA_CONFIG_PATH}'...")
    
    try:
        # 1. Download base model from Hugging Face
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_BASE_MODEL)
        
        # 2. Apply LoRA fine-tuning weights
        # Note: This step assumes the 'lora_config' directory exists in your project root.
        model = PeftModel.from_pretrained(model, LORA_CONFIG_PATH)
        
        # 3. Merge weights and save the full model locally
        model = model.merge_and_unload()
        model.save_pretrained(WHISPER_FT_MODEL_PATH, save_serialization=False)
        
        print(f"Successfully created and saved fine-tuned model to '{WHISPER_FT_MODEL_PATH}'.")
    
    except Exception as e:
        print(f"ERROR: Could not create fine-tuned Whisper model.")
        print(f"Make sure '{LORA_CONFIG_PATH}' exists and contains valid LoRA weights.")
        print(f"Details: {e}")


def cache_demucs_model():
    """
    Downloads the 'mdx_extra' model used by Demucs for source separation.
    Demucs will cache this in the appropriate user directory within the container image.
    """
    print(f"\n--- Caching Demucs Model ---")
    try:
        print("Downloading Demucs model 'mdx_extra'...")
        demucs.pretrained.get_model('mdx_extra')
        print("Demucs model 'mdx_extra' cached successfully.")
    except Exception as e:
        print(f"Error caching Demucs model: {e}")


def cache_gemma_model():
    """
    Downloads the 'google/gemma-2-9b-it' model and tokenizer from Hugging Face
    and saves them to a local directory for fast loading.
    """
    print(f"\n--- Caching Gemma LLM ---")
    
    MODEL_ID = "google/gemma-2-9b-it"
    SAVE_DIRECTORY = "/model-cache/gemma-2-9b-it"

    # Check if the model is already cached
    if os.path.exists(os.path.join(SAVE_DIRECTORY, 'model.safetensors')):
        print(f"Gemma model already found at '{SAVE_DIRECTORY}'. Skipping download.")
        return

    print(f"Downloading Gemma model '{MODEL_ID}'...")
    
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token not found in environment variable HF_TOKEN")

        # Download and save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
        tokenizer.save_pretrained(SAVE_DIRECTORY)
        print("Tokenizer downloaded and saved.")

        # Download and save the model
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=hf_token)
        model.save_pretrained(SAVE_DIRECTORY)
        print(f"Successfully downloaded and saved Gemma model to '{SAVE_DIRECTORY}'.")

    except Exception as e:
        print(f"ERROR: Could not download Gemma model.")
        print(f"Please ensure your HF_TOKEN is valid and you have accepted the terms for {MODEL_ID}.")
        print(f"Details: {e}")


def cache_vad_model():
    """
    Downloads the Silero VAD model from PyTorch Hub to pre-cache it.
    """
    print(f"\n--- Caching Silero VAD Model ---")
    try:
        # This command triggers the download to the torch hub cache
        torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github')
        print("Silero VAD model cached successfully.")
    except Exception as e:
        print(f"Error caching Silero VAD model: {e}")

if __name__ == "__main__":
    cache_whisper_model()
    cache_demucs_model()
    cache_gemma_model()
    cache_vad_model()