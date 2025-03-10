import os
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from train_audio_model import MeowCNN
import logging
import subprocess
import tempfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parameters - these must match the training parameters exactly
TARGET_SR = 16000
DURATION = 1.0     # Use 1.0 to match training
N_MELS = 128
HOP_LENGTH = 512

# Load the processed dataset to get the correct time dimension
processed_data_path = os.path.join("audio_data", "processed", "audio_dataset.npz")
if not os.path.exists(processed_data_path):
    raise ValueError(f"Processed data not found at {processed_data_path}")

data = np.load(processed_data_path, allow_pickle=True)
T_DIM = data["X_train"].shape[2]  # Get the time dimension (32)
logger.info(f"Loaded training data with shape {data['X_train'].shape}")
logger.info(f"Using time dimension: {T_DIM}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install it using: brew install ffmpeg")
        return False

def convert_m4a_to_wav(input_path, output_path):
    """Convert m4a to wav using ffmpeg."""
    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', str(TARGET_SR),
            '-ac', '1',
            '-y',  # Overwrite output file if it exists
            output_path
        ], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {input_path}: {e.stderr.decode()}")
        return False

def validate_bentley_meows():
    """Validate the model specifically on Bentley's meows."""
    if not check_ffmpeg():
        return
    
    # Create temporary directory for WAV files
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # Load the model
        model_path = os.path.join("models", "meow_classifier.pth")
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return
        
        model = MeowCNN(T_DIM).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Create output directory for visualizations
        output_dir = os.path.join("validation_results", "bentley_meows")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get Bentley's meow files
        bentley_dir = os.path.join("data", "raw_audio", "bentley_meows")
        meow_files = [f for f in os.listdir(bentley_dir) if f.endswith('.m4a')]
        
        results = []
        logger.info(f"Testing {len(meow_files)} meow samples from Bentley")
        
        for i, filename in enumerate(meow_files):
            m4a_path = os.path.join(bentley_dir, filename)
            wav_path = os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}.wav")
            
            logger.info(f"\nProcessing file {i+1}/{len(meow_files)}: {filename}")
            
            # Convert M4A to WAV
            if not convert_m4a_to_wav(m4a_path, wav_path):
                continue
            
            # Load and process the WAV file
            try:
                audio, sr = librosa.load(wav_path, sr=TARGET_SR)
                
                # Ensure audio is the correct duration
                target_length = int(TARGET_SR * DURATION)
                if len(audio) > target_length:
                    audio = audio[:target_length]
                else:
                    audio = np.pad(audio, (0, max(0, target_length - len(audio))))
                
                # Convert to mel spectrogram
                S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS,
                                                 hop_length=HOP_LENGTH)
                S_db = librosa.power_to_db(S, ref=np.max)
                S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
                
                # Ensure correct dimensions
                if S_norm.shape[1] != T_DIM:
                    S_norm = librosa.util.fix_length(S_norm, size=T_DIM, axis=1)
                
                # Prepare for model
                spec = np.expand_dims(np.expand_dims(S_norm, axis=0), axis=0)
                spec_tensor = torch.tensor(spec, dtype=torch.float32).to(device)
                
                # Get prediction
                with torch.no_grad():
                    output = model(spec_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    meow_prob = probs[0, 1].item()
                    
                    result = {
                        'filename': filename,
                        'confidence': meow_prob,
                        'prediction': 'MEOW' if meow_prob > 0.60 else 'NOT MEOW'
                    }
                    results.append(result)
                    
                    # Log result
                    logger.info(f"Prediction: {result['prediction']}")
                    logger.info(f"Confidence: {meow_prob*100:.1f}%")
                    
                    # Create visualization
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(1, 2, 1)
                    plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
                    plt.title(f"Waveform: {filename}")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Amplitude")
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(S_norm, aspect='auto', origin='lower')
                    plt.title(f"Mel Spectrogram (Confidence: {meow_prob*100:.1f}%)")
                    plt.colorbar()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"analysis_{i+1}.png"))
                    plt.close()
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        # Summary
        logger.info("\n===== VALIDATION SUMMARY =====")
        successful_detections = sum(1 for r in results if r['prediction'] == 'MEOW')
        logger.info(f"Successfully detected {successful_detections}/{len(results)} meows")
        
        avg_confidence = np.mean([r['confidence'] for r in results]) * 100
        logger.info(f"Average confidence: {avg_confidence:.1f}%")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary files")

if __name__ == "__main__":
    validate_bentley_meows() 