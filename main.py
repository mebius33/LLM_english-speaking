"""
English Speaking Practice System with GPU Acceleration
Features:
1. Record until user stops speaking (silence detection)
2. Whisper transcription (GPU if available)
3. Ollama LLM + Mistral for natural feedback
4. Coqui TTS feedback (GPU if available)
5. Automatic playback of generated audio
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from utils.audio import record_audio
from utils.whisper_model import load_whisper_model, transcribe_audio
from utils.ollama_feedback import ollama_feedback
from utils.tts_model import load_tts_model, tts_speak
import sys
import tomli as tomllib
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

toml_path = Path("config.toml")
with open(toml_path, "rb") as f:
    config = tomllib.load(f)
WHISPER_MODEL = config["models"]["whisper_model"]
OLLAMA_MODEL = config["models"]["ollama_model"]
TTS_MODEL_NAME = config["models"]["tts_model"]

def main():
    # Load models
    whisper_model = load_whisper_model(WHISPER_MODEL)
    tts_model = load_tts_model(TTS_MODEL_NAME)

    print("🟢 English speaking practice started. Press Ctrl+C to stop.")

    try:
        while True:
            # Record user speech until silence is detected
            record_audio()

            # Transcribe audio using Whisper
            transcript = transcribe_audio(whisper_model)
            if not transcript.strip():
                print("⚠️ No speech detected. Please try again.")
                continue

            # Get natural feedback from Ollama
            feedback = ollama_feedback(transcript, OLLAMA_MODEL)

            # Convert feedback to speech and play it
            tts_speak(feedback, tts_model)

            print("🔁 Next round\n")

    except KeyboardInterrupt:
        print("🛑 Practice ended.")

if __name__ == "__main__":
    main()
