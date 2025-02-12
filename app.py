print("Starting cat detection app...")
import cv2
import torch
import torchaudio
import numpy as np
import pyaudio
import pyttsx3
from torchvision import transforms, models
from torch import nn
from gtts import gTTS
import os
import threading
import queue

# Paths to models
CAT_MODEL_PATH = "models/cat_detection_model.pth"
MEOW_MODEL_PATH = "models/meow_detection_model.pth"

# Load the trained cat detection model
def load_cat_model():
    print("Checking if model file exists...")
    if not os.path.exists(CAT_MODEL_PATH):
        print(f"Error: Model file {CAT_MODEL_PATH} does not exist.")
        return None

    print("Loading the model architecture...")
    model = models.resnet50(weights=None)  # Load without pretrained weights
    num_features = model.fc.in_features

    # Match the architecture used during training
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    print("Loading model weights...")
    model.load_state_dict(torch.load(CAT_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    print("Model loaded successfully.")
    return model

# Load the trained meow detection model
def load_meow_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(32 * 16 * 8, 128),  # Match dimensions from training
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1)
    )
    model.load_state_dict(torch.load(MEOW_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

# Preprocessing for image input
def preprocess_image(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(frame).unsqueeze(0)

# Preprocessing for audio input
def preprocess_audio(audio):
    # Convert to tensor and reshape
    waveform = torch.tensor(audio).float().unsqueeze(0)
    
    # Ensure length is 16000 samples (1 second)
    target_length = 16000
    if waveform.size(1) < target_length:
        padding = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :target_length]
    
    # Create mel spectrogram transform with same parameters as training
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64  # Adjust this if model expects a different number of mel bands
    )
    
    # Convert to mel spectrogram
    mel_spectrogram = mel_transform(waveform)
    mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    
    # Ensure the output size matches the model's expected input size
    mel_spectrogram_db = mel_spectrogram_db.unsqueeze(0)  # Add batch dimension

    # Print the shape for debugging
    print("Preprocessed audio shape:", mel_spectrogram_db.shape)

    return mel_spectrogram_db

# Detect cat in image frame
def detect_cat(frame, cat_model):
    original_height, original_width = frame.shape[:2]
    preprocessed_frame = preprocess_image(frame)
    
    # Move the preprocessed frame to the same device as the model
    device = next(cat_model.parameters()).device
    preprocessed_frame = preprocessed_frame.to(device)
    
    with torch.no_grad():
        output = cat_model(preprocessed_frame)
        confidence = torch.sigmoid(output).item()
        is_cat = confidence > 0.7
        
        # For simplicity, let's assume the cat takes up a significant portion of the frame
        if is_cat:
            # Define a box that's 60% of the frame size
            box_width = int(original_width * 0.6)
            box_height = int(original_height * 0.6)
            
            # Calculate box coordinates (centered)
            x1 = (original_width - box_width) // 2
            y1 = (original_height - box_height) // 2
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            return True, confidence, (x1, y1, x2, y2)
    
    return False, 0.0, None

# Detect meow in audio
def detect_meow(audio_data, meow_model):
    preprocessed_audio = preprocess_audio(audio_data)
    with torch.no_grad():
        output = meow_model(preprocessed_audio)
        return torch.sigmoid(output).item() > 0.5

# Trigger sound alert
def play_alert():
    tts = gTTS("Cat detected! Meow detected!", lang='en')
    tts.save("alert.mp3")
    os.system("afplay alert.mp3")  # Use 'afplay' on macOS

# Define the model
def create_meow_detection_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        # Adjusted input size for the linear layer based on our mel spectrogram dimensions
        nn.Linear(32 * 16 * 16, 128),  # Adjust dimensions based on mel spectrogram size
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1)
    )
    return model

# Function to test if an image contains a cat
def test_image_for_cat(image_path, model):
    if model is None:
        print("Error: Cat model is not loaded.")
        return

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Move the model and input to the appropriate device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    image_tensor = image_tensor.to(device)

    # Run the model
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).item()

    # Print the result to the terminal
    if prediction > 0.7:
        print(f"The image at {image_path} is predicted to contain a cat with confidence {prediction:.2f}.")
    else:
        print(f"The image at {image_path} is predicted not to contain a cat with confidence {1 - prediction:.2f}.")

def detect_cat_in_thread(frame_queue, result_queue, cat_model):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        cat_detected, confidence, bbox = detect_cat(frame, cat_model)
        result_queue.put((cat_detected, confidence, bbox))

if __name__ == "__main__":
    # Load models
    print("Loading models...")
    cat_model = load_cat_model()
    meow_model = load_meow_model()

    # Test a specific image
    test_image_path = "tests/random.jpg"
    test_image_for_cat(test_image_path, cat_model)

    # Initialize camera and audio stream
    print("Initializing camera and audio stream...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduce frame height

    p = pyaudio.PyAudio()
    frame_queue = queue.Queue()
    result_queue = queue.Queue()

    detection_thread = threading.Thread(target=detect_cat_in_thread, args=(frame_queue, result_queue, cat_model))
    detection_thread.start()

    try:
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=32000)  # Increased buffer size

        print("Starting detection...")
        
        while True:
            # Read video frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Add frame to queue for processing
            if frame_queue.qsize() < 2:  # Limit queue size to prevent memory issues
                frame_queue.put(frame)

            # Check for detection results
            if not result_queue.empty():
                cat_detected, confidence, bbox = result_queue.get()

                # Display detection results on video feed
                if cat_detected:
                    # Draw bounding box
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Display confidence score
                    confidence_text = f"Cat: {confidence:.2f}"
                    cv2.putText(frame, confidence_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Cat", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow("Cat Detection", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        # Signal the detection thread to exit
        frame_queue.put(None)
        detection_thread.join()

        # Release resources
        cap.release()
        if stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()
        cv2.destroyAllWindows()
