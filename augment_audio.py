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
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, n_steps):
    """Shift the pitch of the audio by n_steps semitones."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def add_noise(audio, noise_factor=0.005):
    """Add random noise to the audio."""
    noise = np.random.randn(len(audio)) * noise_factor
    return audio + noise

def change_volume(audio, factor):
    """Change the volume of the audio."""
    return audio * factor

def apply_filter(audio, sr, filter_type='lowpass', cutoff_freq=1000):
    """Apply a filter to the audio."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype=filter_type)
    return signal.filtfilt(b, a, audio)

def add_reverb(audio, sr, reverberance=50):
    """Add reverb effect to audio using convolution with a simple impulse response."""
    # Create a simple impulse response
    reverb_length = int(sr * reverberance / 1000)  # Convert ms to samples
    impulse_response = np.exp(-np.linspace(0, 5, reverb_length))
    
    # Normalize the impulse response
    impulse_response = impulse_response / np.sum(impulse_response)
    
    # Apply convolution
    return signal.convolve(audio, impulse_response, mode='full')[:len(audio)]

def augment_audio_file(file_path, output_dir, num_augmentations=5, visualize=False):
    """Augment a single audio file and save multiple variations."""
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None)
        
        # Create base filename for augmented files
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # If visualization is enabled, plot the original waveform
        if visualize:
            plt = plot_waveform(audio, sr, f"Original: {base_name}")
            plt.savefig(os.path.join(output_dir, f"{base_name}_original.png"))
            plt.close()
        
        # Generate augmented versions
        for i in range(num_augmentations):
            # Apply a random combination of augmentations
            augmented = audio.copy()
            
            # Randomly select augmentations to apply
            augmentation_types = random.sample([
                'time_stretch', 'pitch_shift', 'add_noise', 
                'change_volume', 'filter', 'reverb'
            ], k=random.randint(2, 4))
            
            augmentation_details = []
            
            if 'time_stretch' in augmentation_types:
                rate = random.uniform(0.8, 1.2)
                augmented = time_stretch(augmented, rate)
                augmentation_details.append(f"time_stretch_{rate:.2f}")
            
            if 'pitch_shift' in augmentation_types:
                n_steps = random.uniform(-3, 3)
                augmented = pitch_shift(augmented, sr, n_steps)
                augmentation_details.append(f"pitch_{n_steps:.1f}")
            
            if 'add_noise' in augmentation_types:
                noise_factor = random.uniform(0.001, 0.01)
                augmented = add_noise(augmented, noise_factor)
                augmentation_details.append(f"noise_{noise_factor:.3f}")
            
            if 'change_volume' in augmentation_types:
                volume_factor = random.uniform(0.7, 1.3)
                augmented = change_volume(augmented, volume_factor)
                augmentation_details.append(f"vol_{volume_factor:.1f}")
            
            if 'filter' in augmentation_types:
                filter_type = random.choice(['lowpass', 'highpass'])
                cutoff = random.randint(500, 4000) if filter_type == 'lowpass' else random.randint(100, 1000)
                augmented = apply_filter(augmented, sr, filter_type, cutoff)
                augmentation_details.append(f"{filter_type}_{cutoff}")
            
            if 'reverb' in augmentation_types:
                reverberance = random.randint(20, 100)
                augmented = add_reverb(augmented, sr, reverberance)
                augmentation_details.append(f"reverb_{reverberance}")
            
            # Create a filename that includes the augmentation details
            aug_filename = f"{base_name}_aug_{i+1}_{'_'.join(augmentation_details)}.wav"
            output_path = os.path.join(output_dir, aug_filename)
            
            # Save the augmented audio
            sf.write(output_path, augmented, sr)
            
            # If visualization is enabled, plot the augmented waveform
            if visualize and i < 3:  # Limit to first 3 augmentations to avoid too many plots
                plt = plot_waveform(augmented, sr, f"Augmented {i+1}: {', '.join(augmentation_details)}")
                plt.savefig(os.path.join(output_dir, f"{base_name}_aug_{i+1}.png"))
                plt.close()
                
        return num_augmentations
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():
    # Define paths
    input_dir = os.path.join("data", "raw_audio", "meows")
    output_dir = os.path.join("data", "raw_audio", "augmented_meows")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio files in the input directory
    audio_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not audio_files:
        print(f"No .wav files found in {input_dir}. Checking for other audio formats...")
        audio_files = glob.glob(os.path.join(input_dir, "*.mp3")) + \
                     glob.glob(os.path.join(input_dir, "*.ogg")) + \
                     glob.glob(os.path.join(input_dir, "*.flac"))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}.")
        return
    
    print(f"Found {len(audio_files)} audio files. Starting augmentation...")
    
    # Set parameters
    num_augmentations_per_file = 10  # Generate 10 variations of each file
    visualize = True  # Set to True to generate waveform visualizations
    
    # Process each audio file
    total_augmentations = 0
    for file_path in tqdm(audio_files, desc="Augmenting audio files"):
        num_created = augment_audio_file(
            file_path, 
            output_dir, 
            num_augmentations=num_augmentations_per_file,
            visualize=visualize
        )
        total_augmentations += num_created
    
    print(f"Augmentation complete! Created {total_augmentations} new audio samples.")
    print(f"Original samples: {len(audio_files)}")
    print(f"Total samples now available: {len(audio_files) + total_augmentations}")
    print(f"Augmented files saved to: {output_dir}")

if __name__ == "__main__":
    main() 