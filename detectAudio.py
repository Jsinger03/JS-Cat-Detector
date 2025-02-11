import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


data_dir = 'data/audio_data'

def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Preprocess audio data
X = []
y = []

for filename in os.listdir(data_dir):
    if filename.endswith('.wav'):
        file_path = os.path.join(data_dir, filename)
        features = extract_features(file_path)
        X.append(features)
        y.append(1)  # 1 for meow

# Add some non-meow sounds (get separate)
non_meow_dir = 'non_meow_sounds'
for filename in os.listdir(non_meow_dir):
    if filename.endswith('.wav'):
        file_path = os.path.join(non_meow_dir, filename)
        features = extract_features(file_path)
        X.append(features)
        y.append(0)  # 0 for non-meow

X = np.array(X)
y = np.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = Sequential([
    Dense(256, activation='relu', input_shape=(13,)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Save the model
model.save('meow_detection_model.h5')
