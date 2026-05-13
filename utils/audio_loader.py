import librosa

def load_audio(file_path, sr=16000):
    waveform, sr = librosa.load(file_path, sr=sr)
    return waveform, sr