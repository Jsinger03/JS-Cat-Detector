# Filename: preprocess_audio.py
import os
import numpy as np
import librosa
import csv
import random

# Parameters
TARGET_SR = 16000  # 16 kHz
DURATION = 1.0     # 1-second clips
N_MELS = 128       # Use 128 mel bands for higher accuracy
HOP_LENGTH = 512

# Directories
POS_DIR = os.path.join("data", "raw_audio", "augmented_meows")
NEG_DIR = os.path.join("data", "raw_audio", "noise")
ESC50_CSV = os.path.join(NEG_DIR, "esc50.csv")  # CSV file with ESC-50 metadata
ESC50_AUDIO_DIR = os.path.join(NEG_DIR, "audio")

# Output directory for processed audio data
OUTPUT_DIR = os.path.join("audio_data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_trim(path, target_sr=TARGET_SR, duration=DURATION):
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
    target_length = int(target_sr * duration)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        if len(y) < target_length:
            padding = target_length - len(y)
            y = np.pad(y, (0, padding))
    return y

def audio_to_mel(y, sr=TARGET_SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def normalize_spectrogram(S_db):
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    return S_norm.astype(np.float32)

def process_audio_file(path):
    y = load_and_trim(path)
    if y is None:
        return None
    S_db = audio_to_mel(y)
    S_norm = normalize_spectrogram(S_db)
    return S_norm

def get_file_list(directory, ext=".wav"):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)]

def load_esc50_negatives():
    negatives = []
    if not os.path.exists(ESC50_CSV):
        print(f"ESC-50 CSV not found at {ESC50_CSV}")
        return negatives
    with open(ESC50_CSV, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Assume CSV has a column "category"; convert to lowercase for comparison
            category = row['category'].strip().lower()
            # Exclude any sample whose category contains "cat"
            if "cat" in category:
                continue
            filename = row['filename'].strip()
            file_path = os.path.join(ESC50_AUDIO_DIR, filename)
            if os.path.exists(file_path):
                negatives.append(file_path)
            else:
                print(f"File {file_path} not found, skipping.")
    return negatives

def train_test_split(file_list, split_ratio=0.8):
    random.shuffle(file_list)
    split_idx = int(len(file_list) * split_ratio)
    return file_list[:split_idx], file_list[split_idx:]

def process_and_collect(file_list):
    data = []
    for path in file_list:
        spec = process_audio_file(path)
        if spec is not None:
            data.append(spec)
    return data

if __name__ == "__main__":
    # Process positive samples (meows)
    pos_files = get_file_list(POS_DIR, ext=".wav")
    print(f"Found {len(pos_files)} positive meow files.")
    pos_train_files, pos_test_files = train_test_split(pos_files, split_ratio=0.8)
    pos_train_data = process_and_collect(pos_train_files)
    pos_test_data = process_and_collect(pos_test_files)
    
    # Process negative samples (from ESC-50, excluding cat samples)
    neg_files = load_esc50_negatives()
    print(f"Found {len(neg_files)} negative (non-meow) files from ESC-50.")
    neg_train_files, neg_test_files = train_test_split(neg_files, split_ratio=0.8)
    neg_train_data = process_and_collect(neg_train_files)
    neg_test_data = process_and_collect(neg_test_files)
    
    # Create labels: 1 for meow, 0 for non-meow
    X_train = np.array(pos_train_data + neg_train_data)
    y_train = np.array([1]*len(pos_train_data) + [0]*len(neg_train_data))
    X_test = np.array(pos_test_data + neg_test_data)
    y_test = np.array([1]*len(pos_test_data) + [0]*len(neg_test_data))
    
    # Shuffle training and testing data
    train_indices = np.arange(len(X_train))
    np.random.shuffle(train_indices)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    test_indices = np.arange(len(X_test))
    np.random.shuffle(test_indices)
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]
    
    # Save the processed dataset as an NPZ file
    output_path = os.path.join(OUTPUT_DIR, "audio_dataset.npz")
    np.savez(output_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Processed audio dataset saved to {output_path}")