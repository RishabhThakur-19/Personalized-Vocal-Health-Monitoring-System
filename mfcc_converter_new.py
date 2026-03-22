# this is new mfcc converter earlier one had a significant error that it was creating mulptiple copies of data for training and testing which during taining phase broke the model



import os
import glob
import librosa
import numpy as np
from tqdm import tqdm

# PATHS


audio_dir = os.path.abspath('./Augmented_data_new')
mfcc_dir = os.path.abspath('./MFCC_features_new')

os.makedirs(mfcc_dir, exist_ok=True)

# GET FILES


wav_files = glob.glob(audio_dir + '/**/*.wav', recursive=True)
print("Total wav files:", len(wav_files))


# PROCESS LOOP


for wav_path in tqdm(wav_files):

    try:
        
        audio, sr = librosa.load(wav_path, sr=22050)

        if len(audio) < 2048:
            continue

       
        audio = np.nan_to_num(audio).astype(np.float32)

   
        # MFCC (UNCHANGED)
   

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13
        )

   
        # SAVE
      

        relative_path = os.path.relpath(wav_path, audio_dir)
        save_path = os.path.join(
            mfcc_dir,
            relative_path.replace(".wav", ".npy")
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # ⚡ faster save
        np.save(save_path, mfcc, allow_pickle=False)

    except Exception as e:
        print("Skipping:", wav_path)
        print(e)

print(" MFCC extraction completed")
