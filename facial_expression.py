import cv2
from deepface import DeepFace

def analyze_face_emotion(frame):
    """Analyze the facial emotions from the given frame."""
    try:
        # Check if the frame is valid
        if frame is None or frame.size == 0:
            print("Empty frame received for emotion analysis.")
            return None
        
        # Analyze emotions in the frame using DeepFace
        print("Analyzing facial emotions...")
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Assuming result is a list of dictionaries
        emotion_scores = result[0]['emotion']  # Get the emotion scores from the first result
        print(f"Emotion analysis result: {emotion_scores}")  # Debug output
        return emotion_scores
    except Exception as e:
        print(f"Error in analyzing face emotion: {e}")
        return None  # Return None if analysis fails
