"""
Coqui TTS utilities:
- Load TTS model
- Convert text to speech and play it
"""

from pathlib import Path
from TTS.api import TTS
import simpleaudio as sa
import os
import time


TTS_OUTPUT_FILE = "feedback.wav"

def load_tts_model(model_name):
    """
    Load TTS model (prefer GPU if available)
    Args:
        model_name (str): Coqui TTS model name
    Returns:
        TTS: Loaded TTS model
    """
    tts = TTS(model_name=model_name, gpu=True)
    print(f"✅ TTS model loaded (GPU)")
    return tts

def tts_speak(text, tts_model, output_file=TTS_OUTPUT_FILE):
    """
    Convert text to speech and play it
    Args:
        text (str): Text to speak
        tts_model (TTS): Loaded TTS model
        output_file (str): Path to save audio
    """
    os.makedirs(Path(output_file).parent, exist_ok=True)
    tts_model.tts_to_file(text=text, file_path=output_file)
    time.sleep(1) 
    print(f"🔊 TTS output saved to {output_file}")
    wave_obj = sa.WaveObject.from_wave_file(output_file)
    play_obj = wave_obj.play()
    play_obj.wait_done()
