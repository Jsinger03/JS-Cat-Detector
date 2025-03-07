# Filename: validate_model.py
import os
import numpy as np
import torch
import librosa
import random
import matplotlib.pyplot as plt
from train_audio_model import MeowCNN
import glob

# Parameters (should match those in preprocess_audio.py)
TARGET_SR = 16000  # 16 kHz
DURATION = 1.0     # 1-second clips
N_MELS = 128       # Use 128 mel bands for higher accuracy
HOP_LENGTH = 512

# Use MPS if available; otherwise, fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def load_and_trim(path, target_sr=TARGET_SR, duration=DURATION):
    """Load and trim audio file to target duration."""
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
    """Convert audio to mel spectrogram."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def normalize_spectrogram(S_db):
    """Normalize spectrogram to [0, 1] range."""
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    return S_norm.astype(np.float32)

def process_audio_file(path):
    """Process audio file to normalized mel spectrogram."""
    y = load_and_trim(path)
    if y is None:
        return None
    S_db = audio_to_mel(y)
    S_norm = normalize_spectrogram(S_db)
    return S_norm

def plot_spectrogram(spec, title="Mel Spectrogram"):
    """Plot a spectrogram."""
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    return plt

def plot_waveform(audio, sr, title="Waveform"):
    """Plot the waveform of an audio signal."""
    plt.figure(figsize=(10, 2))
    plt.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    return plt

def validate_model_on_original_samples():
    """Test the trained model on original meow samples."""
    # Load the trained model
    model_path = os.path.join("models", "meow_classifier.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train the model first.")
        return
    
    # Get the time dimension from a processed sample
    processed_data_path = os.path.join("audio_data", "processed", "audio_dataset.npz")
    if not os.path.exists(processed_data_path):
        print(f"Processed data not found at {processed_data_path}.")
        return
    
    data = np.load(processed_data_path, allow_pickle=True)
    t_dim = data["X_train"].shape[2]  # Get time dimension from processed data
    
    # Initialize model
    model = MeowCNN(t_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get original meow files
    meow_dir = os.path.join("data", "raw_audio", "meows")
    meow_files = glob.glob(os.path.join(meow_dir, "*.wav"))
    
    if not meow_files:
        print(f"No .wav files found in {meow_dir}. Checking for other audio formats...")
        meow_files = glob.glob(os.path.join(meow_dir, "*.mp3")) + \
                    glob.glob(os.path.join(meow_dir, "*.ogg")) + \
                    glob.glob(os.path.join(meow_dir, "*.flac"))
    
    if not meow_files:
        print(f"No audio files found in {meow_dir}.")
        return
    
    # Get some non-meow files for comparison
    neg_dir = os.path.join("data", "raw_audio", "noise", "audio")
    neg_files = []
    if os.path.exists(neg_dir):
        neg_files = glob.glob(os.path.join(neg_dir, "*.wav"))
        if len(neg_files) > 10:
            neg_files = random.sample(neg_files, 10)
    
    # Select random meow files for testing
    num_samples = min(3, len(meow_files))
    test_meow_files = random.sample(meow_files, num_samples)
    
    # Create output directory for visualizations
    output_dir = os.path.join("validation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n===== VALIDATING MODEL ON ORIGINAL MEOW SAMPLES =====")
    
    # Test on meow files
    for i, file_path in enumerate(test_meow_files):
        filename = os.path.basename(file_path)
        print(f"\nTesting file {i+1}/{num_samples}: {filename}")
        
        # Load and process audio
        y = load_and_trim(file_path)
        if y is None:
            print(f"  Error: Could not load audio file.")
            continue
        
        # Plot and save waveform
        plt_wave = plot_waveform(y, TARGET_SR, f"Waveform: {filename}")
        plt_wave.savefig(os.path.join(output_dir, f"waveform_{i+1}.png"))
        plt_wave.close()
        
        # Process to spectrogram
        spec = process_audio_file(file_path)
        if spec is None:
            print(f"  Error: Could not process audio file.")
            continue
        
        # Plot and save spectrogram
        plt_spec = plot_spectrogram(spec, f"Mel Spectrogram: {filename}")
        plt_spec.savefig(os.path.join(output_dir, f"spectrogram_{i+1}.png"))
        plt_spec.close()
        
        # Prepare for model input
        spec = np.expand_dims(spec, axis=0)  # Add batch dimension
        spec = np.expand_dims(spec, axis=0)  # Add channel dimension
        spec_tensor = torch.tensor(spec, dtype=torch.float32).to(device)
        
        # Get model prediction
        with torch.no_grad():
            output = model(spec_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            meow_prob = probabilities[0, 1].item() * 100  # Probability of class 1 (meow)
        
        # Print results
        result = "MEOW" if pred_class == 1 else "NOT MEOW"
        print(f"  Prediction: {result}")
        print(f"  Confidence: {meow_prob:.2f}% chance of being a meow")
    
    # Test on a few non-meow files if available
    if neg_files:
        print("\n===== VALIDATING MODEL ON NON-MEOW SAMPLES =====")
        num_neg_samples = min(2, len(neg_files))
        test_neg_files = random.sample(neg_files, num_neg_samples)
        
        for i, file_path in enumerate(test_neg_files):
            filename = os.path.basename(file_path)
            print(f"\nTesting non-meow file {i+1}/{num_neg_samples}: {filename}")
            
            # Process file
            spec = process_audio_file(file_path)
            if spec is None:
                print(f"  Error: Could not process audio file.")
                continue
            
            # Prepare for model input
            spec = np.expand_dims(spec, axis=0)  # Add batch dimension
            spec = np.expand_dims(spec, axis=0)  # Add channel dimension
            spec_tensor = torch.tensor(spec, dtype=torch.float32).to(device)
            
            # Get model prediction
            with torch.no_grad():
                output = model(spec_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                pred_class = torch.argmax(probabilities, dim=1).item()
                meow_prob = probabilities[0, 1].item() * 100  # Probability of class 1 (meow)
            
            # Print results
            result = "MEOW" if pred_class == 1 else "NOT MEOW"
            print(f"  Prediction: {result}")
            print(f"  Confidence: {meow_prob:.2f}% chance of being a meow")
    
    print(f"\nValidation complete! Visualizations saved to {output_dir}")

def check_train_test_split():
    """Check if augmented meows are properly split between train and test sets."""
    # Load the processed dataset
    processed_data_path = os.path.join("audio_data", "processed", "audio_dataset.npz")
    if not os.path.exists(processed_data_path):
        print(f"Processed data not found at {processed_data_path}. Run preprocess_audio.py first.")
        return
    
    data = np.load(processed_data_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Count positive samples in train and test sets
    train_pos_count = np.sum(y_train == 1)
    test_pos_count = np.sum(y_test == 1)
    train_neg_count = np.sum(y_train == 0)
    test_neg_count = np.sum(y_test == 0)
    
    print("\n===== DATASET SPLIT ANALYSIS =====")
    print(f"Training set: {len(X_train)} samples")
    print(f"  - Positive (meow) samples: {train_pos_count} ({train_pos_count/len(X_train)*100:.1f}%)")
    print(f"  - Negative (non-meow) samples: {train_neg_count} ({train_neg_count/len(X_train)*100:.1f}%)")
    
    print(f"\nTest set: {len(X_test)} samples")
    print(f"  - Positive (meow) samples: {test_pos_count} ({test_pos_count/len(X_test)*100:.1f}%)")
    print(f"  - Negative (non-meow) samples: {test_neg_count} ({test_neg_count/len(X_test)*100:.1f}%)")
    
    # Check if there are positive samples in both train and test sets
    if train_pos_count > 0 and test_pos_count > 0:
        print("\n✅ Both training and test sets contain positive (meow) samples.")
        print(f"   Train/test split ratio for positive samples: {train_pos_count/(train_pos_count+test_pos_count):.2f}/{test_pos_count/(train_pos_count+test_pos_count):.2f}")
    else:
        print("\n❌ WARNING: Either training or test set is missing positive (meow) samples!")
    
    # Check if the split is reasonable
    total_pos = train_pos_count + test_pos_count
    expected_train_ratio = 0.8  # Expected train ratio from preprocess_audio.py
    actual_train_ratio = train_pos_count / total_pos
    
    if abs(actual_train_ratio - expected_train_ratio) < 0.1:  # Within 10% of expected
        print(f"✅ The train/test split ratio is close to the expected {expected_train_ratio:.1f}/{1-expected_train_ratio:.1f} ratio.")
    else:
        print(f"⚠️ The train/test split ratio ({actual_train_ratio:.2f}/{1-actual_train_ratio:.2f}) differs from the expected {expected_train_ratio:.1f}/{1-expected_train_ratio:.1f} ratio.")

if __name__ == "__main__":
    # First check if the train/test split includes augmented meows
    check_train_test_split()
    
    # Then validate the model on original meow samples
    validate_model_on_original_samples() 