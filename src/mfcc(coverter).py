import os
import glob
import librosa
import numpy as np

# Input audio folder (can be original or noisy dataset)
audio_dir = os.path.abspath('./Augmented_data')

# Output MFCC folder
mfcc_dir = os.path.abspath('./MFCC_features')

os.makedirs(mfcc_dir, exist_ok=True)

# Find wav files
wav_files = glob.glob(audio_dir + '/**/*.wav', recursive=True)

print("Total wav files:", len(wav_files))

for wav_path in wav_files:
    
    try:
        # Load audio
        audio, sr = librosa.load(wav_path, sr=22050)
        if len(audio) < 2048:
            print("Skipping short file:", wav_path)
            continue
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13
        )

        # Save file path structure
        relative_path = os.path.relpath(wav_path, audio_dir)
        save_path = os.path.join(mfcc_dir, relative_path.replace(".wav", ".npy"))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save MFCC
        np.save(save_path, mfcc)

    except Exception as e:
        print("Skipping:", wav_path)
        print(e)

print("MFCC extraction complete!")