import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Define a function to calculate body language score based on key landmarks
def calculate_body_language_score(body_landmarks):
    # Example: Calculate distance between shoulders and hips
    shoulder_left = body_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    shoulder_right = body_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    hip_left = body_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    hip_right = body_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Calculate distances
    shoulder_distance = np.linalg.norm(np.array([shoulder_left.x, shoulder_left.y]) - np.array([shoulder_right.x, shoulder_right.y]))
    hip_distance = np.linalg.norm(np.array([hip_left.x, hip_left.y]) - np.array([hip_right.x, hip_right.y]))

    # Example score calculation (this can be adjusted based on your criteria)
    score = shoulder_distance / (hip_distance + 1e-5)  # Adding a small value to avoid division by zero

    # Normalize score to a scale of 0 to 1
    normalized_score = min(max(score / 2, 0), 1)  # Adjust as necessary for your specific use case

    return normalized_score

# Detect body language from video frames
def detect_body_language(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform pose detection
    result = pose.process(rgb_frame)
    
    if result.pose_landmarks:
        body_landmarks = result.pose_landmarks.landmark  # Get the landmark data
        # Calculate and return the body language score
        return calculate_body_language_score(body_landmarks)
    
    return 0  # Default score if no landmarks are found
