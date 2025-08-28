"""
Whisper utilities:
- Load Whisper model
- Transcribe audio
"""

from pathlib import Path
from faster_whisper import WhisperModel
import torch

WHISPER_MODEL = "medium.en"

def load_whisper_model(model_name=WHISPER_MODEL):
    """
    Load Whisper model with GPU if available
    Args:
        model_name (str): Model name
    Returns:
        WhisperModel: Loaded model instance
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device.startswith("cuda") else "float32"
    print(f"⏳ Loading Whisper model '{model_name}' on {device} ...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print("✅ Whisper model loaded")
    return model

def transcribe_audio(model, audio_file="input.wav"):
    """
    Transcribe audio file using Whisper
    Args:
        model (WhisperModel): Loaded Whisper model
        audio_file (str): Path to audio file
    Returns:
        str: Transcribed text
    """
    if not Path(audio_file).exists():
        print(f"❌ Audio file '{audio_file}' not found")
        return ""
    segments, info = model.transcribe(audio_file)
    transcript = " ".join([s.text for s in segments])
    print("🎯 Transcript:", transcript)
    return transcript
