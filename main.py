import cv2
import numpy as np
import os
from video_capture import capture_video_and_audio
from facial_expression import analyze_face_emotion
from body_language import detect_body_language
from audio_analysis import extract_audio_features
from model import predict_lie

def main():
    video_file = "output.avi"
    audio_file = "output.wav"

    # Capture video and audio
    print("Capturing video and audio...")
    capture_video_and_audio(video_output=video_file, audio_output=audio_file)

    # Extract audio features
    print("Extracting audio features...")
    audio_features = extract_audio_features(audio_file)

    # Read the captured video to analyze the first frame
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read the video frame.")
        return

    # Analyze facial emotions
    print("Analyzing facial emotions...")
    facial_emotion = analyze_face_emotion(frame)

    # Debugging: Print the facial emotion results
    print(f"Facial Emotion Analysis Result: {facial_emotion}")

    # Analyze body language
    print("Analyzing body language...")
    body_language_score = detect_body_language(frame)

    # Check if facial emotion data was returned
    if facial_emotion is None:
        print("Facial emotion analysis failed.")
        return

    # Check if body language score is valid
    if body_language_score is None:
        print("Body language analysis failed. Using default score of 0.")
        body_language_score = 0  # Default score if analysis fails

    # Predict lie and get explanation
    print("Predicting lie...")
    lie_prediction, explanation = predict_lie(facial_emotion, body_language_score, audio_features)

    # Print prediction and explanation
    print(f"Prediction: {lie_prediction}")
    print(f"Explanation: {explanation}")

    # Clean up output files
    print("Cleaning up output files...")
    try:
        os.remove(video_file)
        os.remove(audio_file)
    except OSError as e:
        print(f"Error removing files: {e}")

if __name__ == "__main__":
    main()
