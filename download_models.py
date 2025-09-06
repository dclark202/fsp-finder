import os
import torch
import demucs.pretrained
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

# --- Constants from ml_logic.py ---
# Ensure these paths match your project structure.
WHISPER_BASE_MODEL = "openai/whisper-medium.en"
WHISPER_FT_MODEL_PATH = 'whisper-medium-ft'
LORA_CONFIG_PATH = './lora_config'

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
        # Depending on requirements, you might want to raise the exception to stop the build.
        # raise e

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

if __name__ == "__main__":
    cache_whisper_model()
    cache_demucs_model()