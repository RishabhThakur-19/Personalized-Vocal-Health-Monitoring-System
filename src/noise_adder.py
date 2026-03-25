import os
import glob
import numpy as np
import librosa
import soundfile as sf

coswara_data_dir = os.path.abspath('./Extracted_data')
output_base = os.path.abspath('./Augmented_data')

noise_levels = {
    "small_noise": 0.002,
    "medium_noise": 0.005,
    "heavy_noise": 0.01
}

# create output folders
for n in noise_levels:
    os.makedirs(os.path.join(output_base, n), exist_ok=True)

wav_files = glob.glob(coswara_data_dir + '/**/*.wav', recursive=True)

print("Total wav files found:", len(wav_files))

for wav_path in wav_files:

    try:
        audio, sr = librosa.load(wav_path, sr=None)

        # clean audio
        audio = np.nan_to_num(audio)

        for noise_type, strength in noise_levels.items():

            noise = np.random.normal(0, strength, audio.shape)
            noisy_audio = audio + noise

            # clip to valid range
            noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

            relative = os.path.relpath(wav_path, coswara_data_dir)
            save_path = os.path.join(output_base, noise_type, relative)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            sf.write(save_path, noisy_audio, sr)

    except Exception as e:
        print("Skipping file:", wav_path)
        print("Reason:", e)

print("Noise augmentation completed.")