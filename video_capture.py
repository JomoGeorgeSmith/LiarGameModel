import cv2
import pyaudio
import wave
import time

def capture_video_and_audio(video_output, audio_output, duration=10):
    # Video Capture
    video_cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not video_cap.isOpened():
        print("Error: Could not open video capture.")
        return

    # Set video resolution
    video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output, fourcc, 20.0, (640, 480))

    # Audio Capture
    chunk = 2048
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    
    frames = []
    start_time = time.time()

    print("Recording video and audio...")

    while True:
        ret, frame = video_cap.read()
        if not ret:
            print("Failed to capture video frame")
            break
        
        out.write(frame)  # Write frame to video file
        cv2.imshow('Video Capture', frame)

        # Record audio
        try:
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
        except Exception as e:
            print(f"Audio recording error: {e}")
            break

        # Break after the specified duration
        if time.time() - start_time > duration:
            break

    # Release resources
    print("Finished recording.")
    video_cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save audio to file
    wave_file = wave.open(audio_output, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(format))
    wave_file.setframerate(rate)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("Audio saved successfully.")

# Example usage:
# capture_video_and_audio('output.avi', 'output.wav', duration=10)
