# Filename: augment_audio.py
import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import random
import glob
import matplotlib.pyplot as plt
from scipy import signal
from pydub import AudioSegment
import shutil
import gc
import torch  # Add PyTorch for GPU acceleration
import time

# Add a flag to control GPU usage - set to False if it's slower
USE_GPU = True

# Check if MPS (Mac) or CUDA is available
if USE_GPU and torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using Apple Silicon GPU via MPS")
elif USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using NVIDIA GPU via CUDA")
else:
    device = torch.device("cpu")
    print(f"Using CPU - no GPU acceleration available")

def plot_waveform(audio, sr, title="Waveform"):
    """Plot the waveform of an audio signal."""
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    return plt

def time_stretch(audio, rate):
    """Time stretch the audio by a rate."""
    result = librosa.effects.time_stretch(audio, rate=rate)
    return np.ascontiguousarray(result, dtype=np.float32)

def pitch_shift(audio, sr, n_steps):
    """Shift the pitch of the audio by n_steps semitones."""
    result = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    return np.ascontiguousarray(result, dtype=np.float32)

# CPU versions of functions for comparison
def add_noise_cpu(audio, noise_factor=0.005):
    """Add random noise to the audio using CPU."""
    noise = np.random.randn(len(audio)).astype(np.float32) * noise_factor
    return audio + noise

def change_volume_cpu(audio, factor):
    """Change the volume of the audio using CPU."""
    return audio * factor

def add_reverb_cpu(audio, sr, reverberance=50):
    """Add reverb effect to audio using CPU."""
    reverb_length = int(sr * reverberance / 1000)
    impulse_response = np.exp(-np.linspace(0, 5, reverb_length)).astype(np.float32)
    impulse_response = impulse_response / np.sum(impulse_response)
    return signal.convolve(audio, impulse_response, mode='full')[:len(audio)]

# GPU versions
def add_noise(audio, noise_factor=0.005):
    """Add random noise to the audio."""
    if device.type == "cpu":
        return add_noise_cpu(audio, noise_factor)
    
    # GPU version
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    audio_tensor = torch.tensor(audio, device=device, dtype=torch.float32)
    noise = torch.randn(len(audio), device=device, dtype=torch.float32) * noise_factor
    result = audio_tensor + noise
    return result.cpu().numpy()

def change_volume(audio, factor):
    """Change the volume of the audio."""
    if device.type == "cpu":
        return change_volume_cpu(audio, factor)
    
    # GPU version
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    audio_tensor = torch.tensor(audio, device=device, dtype=torch.float32)
    result = audio_tensor * factor
    return result.cpu().numpy()

def apply_filter(audio, sr, filter_type='lowpass', cutoff_freq=1000):
    """Apply a filter to the audio."""
    # Ensure audio is float32
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype=filter_type)
    return signal.filtfilt(b, a, audio)

def add_reverb(audio, sr, reverberance=50):
    """Add reverb effect to audio."""
    if device.type == "cpu":
        return add_reverb_cpu(audio, sr, reverberance)
    
    # GPU version
    reverb_length = int(sr * reverberance / 1000)
    impulse_response = np.exp(-np.linspace(0, 5, reverb_length))
    impulse_response = impulse_response / np.sum(impulse_response)
    
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    impulse_response = np.ascontiguousarray(impulse_response, dtype=np.float32)
    
    audio_tensor = torch.tensor(audio, device=device, dtype=torch.float32)
    impulse_tensor = torch.tensor(impulse_response, device=device, dtype=torch.float32)
    
    result = torch.nn.functional.conv1d(
        audio_tensor.view(1, 1, -1),
        impulse_tensor.view(1, 1, -1),
        padding=len(impulse_response)-1
    ).view(-1)
    
    return result[:len(audio)].cpu().numpy()

def convert_m4a_to_wav(input_path, output_path):
    """Convert m4a file to wav format."""
    try:
        audio = AudioSegment.from_file(input_path, format="m4a")
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False

def augment_audio_file(file_path, output_dir, num_augmentations=5, save_plots=False):
    """Augment a single audio file and save multiple variations."""
    try:
        # Load the audio file with explicit float32 dtype
        audio, sr = librosa.load(file_path, sr=None, dtype=np.float32)
        
        # Ensure audio is contiguous and float32
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        
        # Create base filename for augmented files
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        is_bentley = "bentley" in base_name.lower()  # Check if this is Bentley's meow
        
        # Generate augmented versions
        for i in range(num_augmentations):
            # Apply a random combination of augmentations
            augmented = audio.copy()
            
            # For Bentley's meows, use more conservative augmentation parameters
            if is_bentley:
                augmentation_types = random.sample([
                    'time_stretch', 'pitch_shift', 'add_noise', 
                    'change_volume', 'reverb'
                ], k=random.randint(2, 3))  # Use fewer simultaneous augmentations
            else:
                augmentation_types = random.sample([
                    'time_stretch', 'pitch_shift', 'add_noise', 
                    'change_volume', 'filter', 'reverb'
                ], k=random.randint(2, 4))
            
            augmentation_details = []
            
            if 'time_stretch' in augmentation_types:
                # More conservative time stretching for Bentley's meows
                rate = random.uniform(0.9, 1.1) if is_bentley else random.uniform(0.8, 1.2)
                augmented = time_stretch(augmented, rate)
                augmentation_details.append(f"time_stretch_{rate:.2f}")
            
            if 'pitch_shift' in augmentation_types:
                # More conservative pitch shifting for Bentley's meows
                n_steps = random.uniform(-2, 2) if is_bentley else random.uniform(-3, 3)
                augmented = pitch_shift(augmented, sr, n_steps)
                augmentation_details.append(f"pitch_{n_steps:.1f}")
            
            if 'add_noise' in augmentation_types:
                # Less noise for Bentley's meows
                noise_factor = random.uniform(0.001, 0.005) if is_bentley else random.uniform(0.001, 0.01)
                augmented = add_noise(augmented, noise_factor)
                augmentation_details.append(f"noise_{noise_factor:.3f}")
            
            if 'change_volume' in augmentation_types:
                # More conservative volume changes for Bentley's meows
                volume_factor = random.uniform(0.8, 1.2) if is_bentley else random.uniform(0.7, 1.3)
                augmented = change_volume(augmented, volume_factor)
                augmentation_details.append(f"vol_{volume_factor:.1f}")
            
            if 'filter' in augmentation_types and not is_bentley:
                # Only apply filters to non-Bentley meows
                filter_type = random.choice(['lowpass', 'highpass'])
                cutoff = random.randint(500, 4000) if filter_type == 'lowpass' else random.randint(100, 1000)
                augmented = apply_filter(augmented, sr, filter_type, cutoff)
                augmentation_details.append(f"{filter_type}_{cutoff}")
            
            if 'reverb' in augmentation_types:
                # Less reverb for Bentley's meows
                reverberance = random.randint(20, 60) if is_bentley else random.randint(20, 100)
                augmented = add_reverb(augmented, sr, reverberance)
                augmentation_details.append(f"reverb_{reverberance}")
            
            # Create a filename that includes the augmentation details
            aug_filename = f"{base_name}_aug_{i+1}_{'_'.join(augmentation_details)}.wav"
            output_path = os.path.join(output_dir, aug_filename)
            
            # Save the augmented audio
            sf.write(output_path, augmented, sr)
            
            if save_plots:
                plt = plot_waveform(augmented, sr, f"Augmented {i+1}: {', '.join(augmentation_details)}")
                # Save plot to file instead of keeping in memory
                plt.savefig(os.path.join(output_dir, f"{base_name}_aug_{i+1}.png"))
                plt.close()  # Important: close the plot to free memory
                
        return num_augmentations
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def batch_process_files(file_list, output_dir, num_augmentations, save_plots=False, batch_size=10):
    """Process files in batches to better utilize GPU."""
    total_created = 0
    # Add progress bar back
    for i in tqdm(range(0, len(file_list), batch_size), desc="Processing batches"):
        batch = file_list[i:i+batch_size]
        for file_path in tqdm(batch, desc="Files in batch", leave=False):
            num_created = augment_audio_file(
                file_path,
                output_dir,
                num_augmentations=num_augmentations,
                save_plots=save_plots
            )
            total_created += num_created
        
        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):  # For newer PyTorch versions with MPS
            torch.mps.empty_cache()
        
        # Force garbage collection
        gc.collect()
    
    return total_created

def main():
    # Define paths
    general_meows_dir = os.path.join("data", "raw_audio", "meows")
    bentley_meows_dir = os.path.join("data", "raw_audio", "bentley_meows")
    output_dir = os.path.join("data", "raw_audio", "augmented_meows")
    temp_dir = os.path.join("data", "raw_audio", "temp_wav")
    
    # Create output and temp directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process Bentley's meows first
    bentley_files = glob.glob(os.path.join(bentley_meows_dir, "*.m4a"))
    bentley_wav_files = []
    
    print(f"Found {len(bentley_files)} Bentley meow files. Converting to WAV...")
    
    # Convert m4a files to wav with progress bar
    for m4a_file in tqdm(bentley_files, desc="Converting to WAV"):
        base_name = os.path.splitext(os.path.basename(m4a_file))[0]
        wav_path = os.path.join(temp_dir, f"{base_name}.wav")
        if convert_m4a_to_wav(m4a_file, wav_path):
            bentley_wav_files.append(wav_path)
    
    # Find general meow files
    general_meow_files = glob.glob(os.path.join(general_meows_dir, "*.wav"))
    if not general_meow_files:
        general_meow_files = glob.glob(os.path.join(general_meows_dir, "*.mp3")) + \
                           glob.glob(os.path.join(general_meows_dir, "*.ogg")) + \
                           glob.glob(os.path.join(general_meows_dir, "*.flac"))
    
    print(f"Found {len(general_meow_files)} general meow files.")
    
    # Try both CPU and GPU to see which is faster
    print("\nTesting performance...")
    test_file = bentley_wav_files[0] if bentley_wav_files else general_meow_files[0]
    
    # Test CPU
    global USE_GPU
    USE_GPU = False
    device = torch.device("cpu")
    start_time = time.time()
    augment_audio_file(test_file, output_dir, num_augmentations=1, save_plots=False)
    cpu_time = time.time() - start_time
    print(f"CPU processing time: {cpu_time:.2f} seconds")
    
    # Test GPU
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        USE_GPU = True
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        start_time = time.time()
        augment_audio_file(test_file, output_dir, num_augmentations=1, save_plots=False)
        gpu_time = time.time() - start_time
        print(f"GPU processing time: {gpu_time:.2f} seconds")
        
        # Use the faster option
        USE_GPU = gpu_time < cpu_time
        device = torch.device("mps" if USE_GPU and torch.backends.mps.is_available() else 
                             "cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
        print(f"Using {'GPU' if USE_GPU else 'CPU'} for processing (faster)")
    
    # Augmentation parameters
    bentley_augmentations = 20  # More augmentations for Bentley's meows
    general_augmentations = 5   # Fewer augmentations for general meows
    visualize = True
    
    # Process Bentley's meows with more variations using batch processing
    print("\nAugmenting Bentley's meows...")
    total_bentley_augmentations = batch_process_files(
        bentley_wav_files,
        output_dir,
        num_augmentations=bentley_augmentations,
        save_plots=visualize,
        batch_size=5  # Process 5 files at a time
    )
    
    # Process general meow files in batches
    print("\nAugmenting general meows...")
    total_general_augmentations = batch_process_files(
        general_meow_files,
        output_dir,
        num_augmentations=general_augmentations,
        save_plots=False,
        batch_size=10  # Process 10 files at a time
    )
    
    # Cleanup temporary files
    shutil.rmtree(temp_dir)
    
    print(f"\nAugmentation complete!")
    print(f"Original Bentley samples: {len(bentley_wav_files)}")
    print(f"Original general samples: {len(general_meow_files)}")
    print(f"Total augmented samples created: {total_bentley_augmentations + total_general_augmentations}")
    print(f"Total samples now available: {len(bentley_wav_files) + len(general_meow_files) + total_bentley_augmentations + total_general_augmentations}")
    print(f"Augmented files saved to: {output_dir}")

if __name__ == "__main__":
    main() 