from core.model import load_model
from core.inference import load_class_names, predict
from utils.audio_loader import load_audio

import numpy as np

print("🎧 Loading Audio Sentinel AI...")

# Load model
model = load_model()

# Load labels
class_names = load_class_names(model)

print("✅ Model Loaded")

# Input audio file
file_path = input("Enter audio file path (mp3/wav): ")

# Load audio
waveform, sr = load_audio(file_path)

print("Audio loaded. Sample rate:", sr)

# Predict
label, confidence, scores = predict(model, waveform, class_names)

print("\n🎯 RESULT")
print("----------------")
print("Detected Sound:", label)
print("Confidence:", confidence)

# Top 5 predictions
print("\n🔍 Top 5 Sounds:\n")

top_indices = np.argsort(scores)[-5:][::-1]

for i in top_indices:
    print(class_names[i], ":", float(scores[i]))