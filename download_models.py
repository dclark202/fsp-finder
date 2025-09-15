import os
import torch
import demucs.pretrained
from transformers import WhisperForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
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
    if os.path.exists(os.path.join(WHISPER_FT_MODEL_PATH, 'pytorch_model.bin')) or \
       os.path.exists(os.path.join(WHISPER_FT_MODEL_PATH, 'model.safetensors')):
        print(f"Fine-tuned model already found at '{WHISPER_FT_MODEL_PATH}'. Skipping creation.")
        return

    print(f"Creating fine-tuned model from base '{WHISPER_BASE_MODEL}' and LoRA config '{LORA_CONFIG_PATH}'...")
    
    try:
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_BASE_MODEL)
        
        # Apply LoRA fine-tuning weights
        model = PeftModel.from_pretrained(model, LORA_CONFIG_PATH)
        
        # Merge weights and save the full model locally
        model = model.merge_and_unload()
        model.save_pretrained(WHISPER_FT_MODEL_PATH, save_serialization=False)
        
        print(f"Successfully created and saved fine-tuned model to '{WHISPER_FT_MODEL_PATH}'.")
    
    except Exception as e:
        print(f"ERROR: Could not create fine-tuned Whisper model.")
        print(f"Make sure '{LORA_CONFIG_PATH}' exists and contains valid LoRA weights.")
        print(f"Details: {e}")

    # Clear the cache to not overload VRAM
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

    # Clear the cache to not overload VRAM
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cache_gemma_model():
    print(f"\n--- Caching Gemma LLM ---")
    
    MODEL_ID = "google/gemma-2-9b-it"
    SAVE_DIRECTORY = "/model-cache/google/gemma-2-9b-it" # MODIFIED THIS PATH

    os.makedirs(SAVE_DIRECTORY, exist_ok=True) # ADDED THIS LINE

    if os.path.exists(os.path.join(SAVE_DIRECTORY, 'model.safetensors.index.json')):
        print(f"Gemma model already found at '{SAVE_DIRECTORY}'. Skipping download.")
        return

    print(f"Downloading Gemma model '{MODEL_ID}' files...")
    
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables.")

        # Define the quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load the model from the Hub with quantization applied
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=hf_token,
            quantization_config=quantization_config,
            device_map={"": "cpu"} # Load to CPU to save GPU VRAM during build
        )

        # Also download the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)

        # Save the quantized model and the tokenizer to the target directory
        model.save_pretrained(SAVE_DIRECTORY)
        tokenizer.save_pretrained(SAVE_DIRECTORY)
        
        print(f"Successfully saved quantized Gemma model to '{SAVE_DIRECTORY}'.")

    except Exception as e:
        print(f"ERROR: Could not download or quantize Gemma model: {e}")

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    cache_whisper_model()
    cache_demucs_model()
    cache_gemma_model()
    cache_vad_model()