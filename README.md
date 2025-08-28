# English Speaking Practice

A local English speaking practice system using:

- Whisper (Faster Whisper) for speech recognition
- Ollama LLM for natural feedback
- Coqui TTS for speech output

## Features

1. Record until user stops speaking (silence detection)
2. Whisper transcription (GPU if available)
3. Natural, conversational feedback from Ollama
4. Coqui TTS feedback (GPU if available)
5. Automatic playback of generated audio

## Requirements

- Python 3.10+
- Conda (optional) or pip
- Microphone and speakers

## Installation

```bash
# using conda
conda env create -f environment.yml
conda activate llm_english-speaking
