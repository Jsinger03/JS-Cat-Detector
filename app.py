import cv2
import numpy as np
import pyaudio
import wave
import librosa
import tensorflow as tf
import pyttsx3

# Load the trained models
cat_model = tf.keras.models.load_model('cat_detection_model.h5')
meow_model = tf.keras.models.load_model('meow_detection_model.h5')

def preprocess_image(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_cat(frame):
    preprocessed_frame = preprocess_image(frame)
    prediction = cat_model.predict(preprocessed_frame)
    return prediction[0][0] > 0.5  # Adjust threshold as needed

def extract_audio_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def detect_meow(audio):
    features = extract_audio_features(audio)
    features = np.expand_dims(features, axis=0)
    prediction = meow_model.predict(features)
    return prediction[0][0] > 0.5  # Adjust threshold as needed

def play_alert():
    engine = pyttsx3.init()
    engine.say("Yeet")
    engine.runAndWait()

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize audio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=22050, input=True, frames_per_buffer=1024)

while True:
    ret, frame = cap.read()
    audio_data = np.frombuffer(stream.read(1024), dtype=np.float32)
    
    if detect_cat(frame) and detect_meow(audio_data):
        play_alert()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
stream.stop_stream()
stream.close()
p.terminate()
cv2.destroyAllWindows()
