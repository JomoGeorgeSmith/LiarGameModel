import cv2
import numpy as np
import librosa
from deepface import DeepFace  # Import DeepFace for emotion detection

# Define score descriptions for body language
body_language_descriptions = {
    (0.0, 0.2): "Very negative body language, indicating discomfort.",
    (0.2, 0.4): "Negative body language, showing signs of unease.",
    (0.4, 0.6): "Neutral body language, indicating no strong feelings.",
    (0.6, 0.8): "Positive body language, showing confidence.",
    (0.8, 1.0): "Very positive body language, indicating openness."
}

# Define score descriptions for audio features
audio_feature_descriptions = {
    (0.0, 0.2): "Low energy in voice, possibly indicating lack of confidence.",
    (0.2, 0.4): "Moderate energy, but still lacks assertiveness.",
    (0.4, 0.6): "Neutral energy in voice, indicating comfort.",
    (0.6, 0.8): "High energy in voice, suggesting confidence and truthfulness.",
    (0.8, 1.0): "Very high energy, indicating excitement or truthfulness."
}

def get_description(score, description_map):
    """Get the description for a given score based on the provided description map."""
    for score_range, description in description_map.items():
        if score_range[0] <= score < score_range[1]:
            return description
    return "Unknown score range"

def analyze_face_emotion(frame):
    try:
        # Check if the frame is valid
        if frame is None or frame.size == 0:
            print("Empty frame received for emotion analysis.")
            return None
        
        # Analyze emotions in the frame using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Assuming result is a list of dictionaries
        emotion_scores = result[0]['emotion']  # Get the emotion scores from the first result
        return emotion_scores
    except Exception as e:
        print(f"Error in analyzing face emotion: {e}")
        return None  # Return None if analysis fails

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


def predict_lie(facial_emotion, body_language_score, audio_features):
    # Ensure facial_emotion is a dictionary with valid emotions
    if not isinstance(facial_emotion, dict) or not facial_emotion:
        return "Unknown", "Facial emotion analysis failed. No data available."

    # Make sure body language score and audio features are valid numbers
    body_language_score = body_language_score if isinstance(body_language_score, (float, int)) else 0
    audio_features = audio_features if isinstance(audio_features, (float, int)) else 0

    # Extract relevant emotion scores
    angry_score = facial_emotion.get('angry', 0)
    disgust_score = facial_emotion.get('disgust', 0)
    fear_score = facial_emotion.get('fear', 0)
    happy_score = facial_emotion.get('happy', 0)
    sad_score = facial_emotion.get('sad', 0)
    surprise_score = facial_emotion.get('surprise', 0)
    neutral_score = facial_emotion.get('neutral', 0)

    # Normalize emotion scores to a 0-1 scale
    total_score = sum(facial_emotion.values())
    normalized_emotions = {emotion: score / total_score if total_score > 0 else 0 for emotion, score in facial_emotion.items()}

    # Calculate a comprehensive facial emotion score
    facial_emotion_score = (
        normalized_emotions['happy'] * 0.5 +  # Positive emotion
        normalized_emotions['neutral'] * 0.3 +  # Neutral emotion
        normalized_emotions['sad'] * -0.4 +  # Negative emotion
        normalized_emotions['angry'] * -0.5 +  # Negative emotion
        normalized_emotions['fear'] * -0.3 +  # Negative emotion
        normalized_emotions['disgust'] * -0.4 +  # Negative emotion
        normalized_emotions['surprise'] * 0.1   # Minor positive emotion
    )

    # Debugging: Print scores and their types
    print(f"Facial Emotion Score: {facial_emotion_score:.2f} (Type: {type(facial_emotion_score)})")
    print(f"Body Language Score: {body_language_score:.2f} (Type: {type(body_language_score)})")
    print(f"Audio Features: {audio_features:.2f} (Type: {type(audio_features)})")

    # Calculate final score for lie prediction
    final_score = (facial_emotion_score * 0.4 + body_language_score * 0.4 + audio_features * 0.2)

    # Determine lie/truth based on final score
    if final_score > 0.5:
        explanation = (
            f"Prediction: Lie. "
            f"Facial emotion score: {facial_emotion_score:.2f}, "
            f"Body language score: {body_language_score:.2f}, "
            f"Audio features score: {audio_features:.2f}."
        )
        return "Lie", explanation
    else:
        explanation = (
            f"Prediction: Truth. "
            f"Facial emotion score: {facial_emotion_score:.2f}, "
            f"Body language score: {body_language_score:.2f}, "
            f"Audio features score: {audio_features:.2f}."
        )
        return "Truth", explanation
