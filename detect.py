# Filename: detect.py
import cv2
import numpy as np
import time
import threading
import queue
import os
import torch
import librosa
import sounddevice as sd
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque
# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the YOLO model (Ultralytics YOLOv8)
try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics YOLO not found. Please install with: pip install ultralytics")
    exit(1)

# Load the trained cat detector model
MODEL_PATH = os.path.join("models", "cat_detector.pt")
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model not found at {MODEL_PATH}. Please train the model or update MODEL_PATH.")
    exit(1)
cat_model = YOLO(MODEL_PATH)
cat_model.fuse()  # fuse model layers for faster inference (optional)
cat_model.to("mps")  # use Apple GPU for inference (if available, otherwise use .to("cpu"))

# Load the trained audio model
AUDIO_MODEL_PATH = os.path.join("models", "meow_classifier.pth")
if not os.path.exists(AUDIO_MODEL_PATH):
    logger.error(f"Audio model not found at {AUDIO_MODEL_PATH}. Please train the audio model first.")
    exit(1)

# Use MPS if available; otherwise, fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define the MeowCNN model (same as in train_audio_model.py)
class MeowCNN(torch.nn.Module):
    def __init__(self, t_dim):
        super(MeowCNN, self).__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool = torch.nn.MaxPool2d(2, 2)
        
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        
        # After 3 poolings, height becomes 128/8 = 16; width becomes t_dim/8.
        self.fc1 = torch.nn.Linear(64 * 16 * (t_dim // 8), 64)
        self.fc2 = torch.nn.Linear(64, 2)  # Two classes: 0 (non-meow), 1 (meow)

    def forward(self, x):
        x = self.dropout(self.pool(torch.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(torch.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(torch.relu(self.bn3(self.conv3(x)))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Audio processing parameters
TARGET_SR = 16000  # 16 kHz
DURATION = 1.0  # Change back to 1.0 to match training
OVERLAP = 0.5  # Adjust overlap accordingly (half of duration)
BUFFER_SIZE = int(TARGET_SR * DURATION)
OVERLAP_SIZE = int(TARGET_SR * OVERLAP)
CONSECUTIVE_DETECTIONS_REQUIRED = 2
DETECTION_WINDOW = 1.0  # Time window to check for consecutive detections

# Add this after your audio processing parameters
N_MELS = 128  # Number of mel bands
HOP_LENGTH = 512
# Calculate the time dimension based on your audio parameters
T_DIM = 1 + int(BUFFER_SIZE / HOP_LENGTH)  # Time dimension of the spectrogram

# Initialize the audio model
audio_model = MeowCNN(T_DIM).to(device)
audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
audio_model.eval()  # Set to evaluation mode
logger.info("Audio model loaded successfully")

# Audio processing functions
def audio_to_mel(y, sr=TARGET_SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    """Convert audio to mel spectrogram."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def normalize_spectrogram(S_db):
    """Normalize spectrogram to [0, 1] range."""
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    return S_norm.astype(np.float32)

# Audio buffer and processing
audio_buffer = np.zeros(int(TARGET_SR * DURATION), dtype=np.float32)
buffer_index = 0
meow_detected = False
meow_confidence = 0.0
meow_lock = threading.Lock()
is_shutting_down = False  # Flag to indicate program is shutting down

@dataclass
class Detection:
    timestamp: float
    confidence: float

recent_detections: Deque[Detection] = deque(maxlen=10)

def check_consecutive_detections(confidence_threshold=0.95, time_window=DETECTION_WINDOW):
    """Check if we have enough recent high-confidence detections."""
    if len(recent_detections) < CONSECUTIVE_DETECTIONS_REQUIRED:
        return False
    
    # Get recent detections within our time window
    current_time = time.time()
    valid_detections = [d for d in recent_detections 
                       if current_time - d.timestamp < time_window 
                       and d.confidence > confidence_threshold]
    
    return len(valid_detections) >= CONSECUTIVE_DETECTIONS_REQUIRED

def audio_callback(indata, frames, time_info, status):
    """Callback function for continuous audio capture."""
    global audio_buffer, buffer_index, meow_detected, meow_confidence, is_shutting_down
    
    if is_shutting_down:
        return
    
    if status:
        logger.warning(f"Audio callback status: {status}")
    
    # Copy data to buffer
    samples_to_copy = min(len(indata), BUFFER_SIZE - buffer_index)
    audio_buffer[buffer_index:buffer_index + samples_to_copy] = indata[:samples_to_copy, 0]
    buffer_index += samples_to_copy
    
    # If buffer is full, process it
    if buffer_index >= BUFFER_SIZE:
        if not is_shutting_down:
            # Process immediately in the callback for lower latency
            process_audio_buffer()
        
        # Shift buffer by overlap amount
        audio_buffer[:OVERLAP_SIZE] = audio_buffer[BUFFER_SIZE-OVERLAP_SIZE:BUFFER_SIZE]
        buffer_index = OVERLAP_SIZE

def process_audio_buffer():
    """Process the audio buffer and detect meows."""
    global meow_detected, meow_confidence
    
    try:
        # Quick check for silence or very low volume
        if np.max(np.abs(audio_buffer)) < 0.01:
            return
        
        # Convert to mel spectrogram
        S_db = audio_to_mel(audio_buffer)
        S_norm = normalize_spectrogram(S_db)
        
        # Prepare for model input
        spec = np.expand_dims(np.expand_dims(S_norm, axis=0), axis=0)
        spec_tensor = torch.tensor(spec, dtype=torch.float32).to(device)
        
        # Get model prediction
        with torch.no_grad():
            output = audio_model(spec_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            meow_prob = probabilities[0, 1].item()
        
        # Add to recent detections if probability is high enough
        if meow_prob > 0.7:  # Lower threshold for tracking
            recent_detections.append(Detection(time.time(), meow_prob))
        
        # Update detection status based on consecutive detections
        with meow_lock:
            meow_detected = check_consecutive_detections()
            meow_confidence = meow_prob
            
            if meow_detected:
                logger.info(f"Meow detected with confidence: {meow_prob:.4f}")
            
    except Exception as e:
        logger.error(f"Error processing audio: {e}")

# Start audio stream in a separate thread
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=TARGET_SR, blocksize=4096)
stream.start()
logger.info("Audio stream started...")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Error: Could not open webcam. Make sure the camera is accessible.")
    stream.stop()
    exit(1)

# Optionally, reduce resolution to improve speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

logger.info("Starting video stream... Press 'q' to quit.")
last_alert_time = 0.0
alert_cooldown = 2.0  # seconds to wait after an alert to avoid spamming

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to capture frame from camera")
            break

        # Run YOLO model on the frame
        results = cat_model.predict(frame, device="mps", verbose=False)
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
                    logger.debug(f"Cat detected with confidence: {conf:.4f}")
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # convert to int
                    # Draw bounding box and label on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Cat: {conf*100:.1f}%"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0,255,0), 2)
        
        # Check audio detection status
        with meow_lock:
            meow_now = meow_detected
            meow_conf = meow_confidence
            meow_detected = False  # Reset flag after checking
        
        # Display meow detection status on frame
        if meow_now:
            meow_text = f"Meow detected: {meow_conf*100:.1f}%"
            cv2.putText(frame, meow_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
        
        # If both cat and meow are detected simultaneously
        current_time = time.time()
        if cat_present and meow_now and current_time - last_alert_time > alert_cooldown:
            # Print alert with confidence scores
            logger.info(f"ALERT: Cat detected (confidence {cat_conf*100:.1f}%) AND meow detected (confidence {meow_conf*100:.1f}%)!")
            # Text-to-speech alert
            os.system("say 'The cat wants to enter'")
            last_alert_time = current_time
            
            # Draw alert on frame
            cv2.putText(frame, "ALERT: Cat wants to enter!", (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the video feed with bounding box (if any)
        cv2.imshow("Cat Detection - Press 'q' to exit", frame)
        # Break on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Quit key pressed, shutting down...")
            break

except KeyboardInterrupt:
    logger.info("Keyboard interrupt detected, shutting down...")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
finally:
    # Set shutdown flag before cleanup
    logger.info("Setting shutdown flag...")
    is_shutting_down = True
    
    # Wait a moment for threads to notice the shutdown flag
    time.sleep(0.5)
    
    # Cleanup
    logger.info("Releasing camera...")
    cap.release()
    logger.info("Closing windows...")
    cv2.destroyAllWindows()
    logger.info("Stopping audio stream...")
    stream.stop()
    logger.info("Detection stopped.")