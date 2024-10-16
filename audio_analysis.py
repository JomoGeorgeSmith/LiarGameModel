# audio_analysis.py
import librosa
import numpy as np

def extract_audio_features(audio_file):
    """Extract audio features from the audio file using librosa."""
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)
        # Extract features (e.g., energy)
        energy = np.mean(librosa.feature.rms(y=y))
        # Normalize the energy to be between 0 and 1
        normalized_energy = min(max(energy / 0.5, 0), 1)  # Adjust as necessary
        return normalized_energy
    except Exception as e:
        print(f"Error in extracting audio features: {e}")
        return 0  # Default score if error occurs
