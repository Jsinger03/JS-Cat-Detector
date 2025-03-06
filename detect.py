# Filename: detect.py
import cv2
import numpy as np
import time
import threading
import queue
import os

# Import the YOLO model (Ultralytics YOLOv8)
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics YOLO not found. Please install with: pip install ultralytics")
    exit(1)

# Load the trained cat detector model
MODEL_PATH = os.path.join("models", "cat_detector.pt")
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}. Please train the model or update MODEL_PATH.")
    exit(1)
model = YOLO(MODEL_PATH)
model.fuse()  # fuse model layers for faster inference (optional)
model.to("mps")  # use Apple GPU for inference (if available, otherwise use .to("cpu"))

# Configure audio capture for meow detection
import sounddevice as sd

audio_queue = queue.Queue()
meow_detected_flag = False

def audio_callback(indata, frames, time_info, status):
    """Callback function for continuous audio capture."""
    global meow_detected_flag
    if status:
        print(f"Audio callback status: {status}", flush=True)
    # Compute volume
    volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
    # If volume above threshold, analyze frequency content
    if volume_norm > 0.01:  # threshold - adjust based on environment
        # Compute FFT of audio chunk to check dominant frequency
        fft = np.fft.rfft(indata[:,0])  # one channel FFT
        freqs = np.fft.rfftfreq(len(indata), d=1.0/16000)
        freq_mag = np.abs(fft)
        if len(freq_mag) > 0:
            dom_freq = freqs[np.argmax(freq_mag)]
        else:
            dom_freq = 0
        # Simple heuristic: consider it a meow if dominant freq is within a range (e.g., 300-600 Hz)
        if 300 < dom_freq < 600:
            # Set a flag or enqueue an event that a meow was detected
            meow_detected_flag = True

# Start audio stream in a separate thread
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
stream.start()
print("Audio stream started...")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Make sure the camera is accessible.")
    stream.stop()
    exit(1)

# Optionally, reduce resolution to improve speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting video stream... Press 'q' to quit.")
last_meow_time = 0.0
alert_cooldown = 2.0  # seconds to wait after an alert to avoid spamming

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame
    results = model.predict(frame, device="mps", verbose=False)
    cat_present = False
    cat_conf = 0.0
    # YOLOv8 results: we iterate through detections
    for r in results:
        for box in r.boxes:  # each bounding box
            cls = int(box.cls[0])  # class id
            conf = float(box.conf[0])
            if cls == 0:  # 'cat' class
                cat_present = True
                cat_conf = conf
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # convert to int
                # Draw bounding box and label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Cat: {conf*100:.1f}%"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0,255,0), 2)
    # Check audio flag
    meow_now = meow_detected_flag
    if meow_now:
        # reset flag immediately to avoid continuous true (we handle one detection at a time)
        meow_detected_flag = False

    # If both cat and meow are detected simultaneously
    current_time = time.time()
    if cat_present and meow_now:
        if current_time - last_meow_time > alert_cooldown:
            # Print alert with confidence scores
            print(f"ALERT: Cat detected (confidence {cat_conf*100:.1f}%) AND meow detected!", flush=True)
            # Text-to-speech alert (using macOS 'say' command)
            os.system("say 'Alert: cat meowing detected'")
            last_meow_time = current_time

    # Display the video feed with bounding box (if any)
    cv2.imshow("Cat Detection - Press 'q' to exit", frame)
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
stream.stop()