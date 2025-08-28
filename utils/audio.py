"""
Audio utilities:
- Record until silence is detected
- Play audio using simpleaudio
"""

import sounddevice as sd
from scipy.io.wavfile import write
import simpleaudio as sa
import numpy as np
from pathlib import Path

# Audio configuration
SAMPLERATE = 16000          # Sampling rate in Hz
AUDIO_FILE = "input.wav"    # Default audio file name
SILENCE_THRESHOLD = 500     # Threshold to consider as silence
SILENCE_DURATION = 2.0      # Duration of silence (seconds) to stop recording
CHUNK_SIZE = 1024           # Audio chunk size for streaming

def record_audio(filename=AUDIO_FILE):
    """
    Record audio from microphone until silence is detected.
    Args:
        filename (str): File to save the recorded audio
    Returns:
        str: Path to the saved audio file
    """
    print("🎤 Recording... Speak now.")
    buffer = []
    silent_chunks = 0
    max_silent_chunks = int(SILENCE_DURATION * SAMPLERATE / CHUNK_SIZE)

    with sd.InputStream(channels=1, samplerate=SAMPLERATE,
                        blocksize=CHUNK_SIZE, dtype='int16') as stream:
        while True:
            data, overflowed = stream.read(CHUNK_SIZE)
            if overflowed:
                print("⚠️ Buffer overflow!")
            buffer.append(data)

            # Check for silence
            if np.max(np.abs(data)) < SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks >= max_silent_chunks:
                print(f"⚠️ No voice detected for {SILENCE_DURATION} seconds. Stopping recording.")
                break

    # Concatenate chunks and save to WAV file
    audio_data = np.concatenate(buffer, axis=0)
    write(filename, SAMPLERATE, audio_data)
    print(f"✅ Audio saved to {filename}")
    return filename

def play_audio(filename):
    """
    Play an audio file using simpleaudio
    Args:
        filename (str): Path to WAV file
    """
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()
