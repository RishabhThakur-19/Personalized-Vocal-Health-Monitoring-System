import os
import numpy as np
import pandas as pd
import glob

# =========================
# PATHS
# =========================

mfcc_folder = "./MFCC_features"
labels_csv = "./Coswara-Data/combined_data.csv"

# =========================
# LOAD LABELS
# =========================

labels_df = pd.read_csv(labels_csv)

label_dict = {}

for _, row in labels_df.iterrows():

    user_id = row["id"]
    status = str(row["covid_status"]).lower()

    if "healthy" in status:
         label = 0
    elif "positive" in status:
        label = 1
    else:
        continue

    label_dict[user_id] = label

print("Total labeled users:", len(label_dict))

# =========================
# FIND MFCC FILES
# =========================

files = glob.glob(mfcc_folder + "/**/*.npy", recursive=True)

print("MFCC files found:", len(files))

X = []
y = []

max_len = 120

# =========================
# PROCESS FILES
# =========================

for file in files:

    try:

        # extract user id from folder
        user_id = os.path.basename(os.path.dirname(file))

        if user_id not in label_dict:
            continue

        mfcc = np.load(file)

        # pad or trim MFCC
        if mfcc.shape[1] < max_len:
            pad = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode="constant")
        else:
            mfcc = mfcc[:, :max_len]

        X.append(mfcc)
        y.append(label_dict[user_id])

    except:
        print("Skipping:", file)

X = np.array(X)
y = np.array(y)

print("Final dataset shape:")
print("X:", X.shape)
print("y:", y.shape)

np.save("X_new.npy", X)
np.save("y_new.npy", y)

print("Dataset saved successfully!")