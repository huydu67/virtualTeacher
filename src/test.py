import os
import keyboard  # For hotkeys
import sounddevice as sd
import numpy as np
import wave

import torch
from faster_whisper import WhisperModel
import time

# Settings
SAMPLE_RATE = 16000  # Whisper works best with 16kHz
CHANNELS = 1
audio_file = r"D:\ML\virtual_teacher\data\voice_test.mp3"

# Load Whisper Model
model_size = "large-v3"
device = "cpu"  # Change to "cuda" if using GPU
compute_type = "int8"

model_path = os.path.join(os.getcwd(), "models", f"whisper-{model_size}")
if not os.path.isdir(model_path):
    model_path = model_size  # Use direct model name if the folder doesn't exist

print(f"Loading model: {model_path}...")
model = WhisperModel(model_path, device=device, compute_type=compute_type)


def transcribe_audio(file_path):
    """ Transcribe audio using Whisper """
    segments, info = model.transcribe(file_path, beam_size=5, word_timestamps=True, vad_filter=True)

    print(f"Detected language: {info.language} (Confidence: {info.language_probability:.2f})")
    print("Transcription:")
    for segment in segments:
        print(segment.text)


# Check if file exists before processing
if os.path.exists(audio_file):
    transcribe_audio(audio_file)
else:
    print(f"Error: File '{audio_file}' not found. Please check the path.")