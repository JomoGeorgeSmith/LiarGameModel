# data_preparation.py
import librosa
import numpy as np

def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)

    # Estimate the pitch using the YIN algorithm
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Get the mean pitch from the detected pitches
    pitch_values = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()  # Get the index of the maximum magnitude
        pitch = pitches[index, i]           # Get the corresponding pitch
        if pitch > 0:                       # Only consider positive pitch values
            pitch_values.append(pitch)

    mean_pitch = np.mean(pitch_values) if pitch_values else 0
    return mean_pitch  # Return the average pitch
