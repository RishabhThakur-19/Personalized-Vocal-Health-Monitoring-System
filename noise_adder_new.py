import os
import glob
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# PATHS


input_dir = "./Extracted_data"
output_dir = "./Augmented_data_new"

noise_levels = {
    "small": 0.002,
    "medium": 0.005,
    "heavy": 0.01
}

# CREATE FOLDERS


for split in ["train", "val"]:
    os.makedirs(os.path.join(output_dir, split, "original"), exist_ok=True)

for level in noise_levels:
    os.makedirs(os.path.join(output_dir, "train", level), exist_ok=True)


# SPLIT USERS (NO LEAKAGE)


users = os.listdir(input_dir)
train_users, val_users = train_test_split(users, test_size=0.2, random_state=42)

print("Train users:", len(train_users))
print("Val users:", len(val_users))


# GET FILES


wav_files = glob.glob(input_dir + "/**/*.wav", recursive=True)
print("Total files:", len(wav_files))


# AUGMENTATION LOOP


for wav_path in tqdm(wav_files):

    try:
        # 
        audio, sr = librosa.load(wav_path, sr=None)

        if len(audio) == 0:
            continue

        audio = np.nan_to_num(audio).astype(np.float32)

        relative = os.path.relpath(wav_path, input_dir)
        user_id = relative.split(os.sep)[0]

        
        # VALIDATION (NO AUGMENT)
        
        if user_id in val_users:
            save_path = os.path.join(output_dir, "val", "original", relative)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            sf.write(save_path, audio, sr)
            continue

        #
        # TRAIN ORIGINAL
        
        save_path = os.path.join(output_dir, "train", "original", relative)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sf.write(save_path, audio, sr)

        
        # NOISE AUGMENTATION
        

        for level_name, strength in noise_levels.items():

            # fast noise generation
            noise = np.random.normal(0, strength, audio.shape).astype(np.float32)

            noisy_audio = audio + noise
            noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

            save_path = os.path.join(output_dir, "train", level_name, relative)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            sf.write(save_path, noisy_audio, sr)

    except Exception as e:
        print("Error:", wav_path, e)

print(" Augmentation completed")