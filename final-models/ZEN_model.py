# COVID COUGH DETECTION MODEL USING CNN + GRU
# DEEP LEARNING PIPELINE FOR BINARY CLASSIFICATION WITH CLASS BALANCING
 
# IMPORT ALL NECESSARY LIBRARIES
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
 
# LOAD FEATURES AND LABELS FROM NUMPY BINARY FILES
print("LOADING DATASET...")
X = np.load("X_new.npy")
y = np.load("y_new.npy")
print("ORIGINAL DATASET SHAPE:", X.shape)
 
# NORMALIZE DATA USING ZERO MEAN AND UNIT VARIANCE (Z-SCORE NORMALIZATION)
print("NORMALIZING DATA...")
X = (X - np.mean(X)) / np.std(X)
 
# ADD A CHANNEL DIMENSION TO MAKE DATA COMPATIBLE WITH CONV2D (HEIGHT, WIDTH, CHANNELS)
X = X[..., np.newaxis]
print("NEW DATASET SHAPE AFTER ADDING CHANNEL DIMENSION:", X.shape)
 
# SPLIT DATA INTO TRAINING AND TEST SETS WITH STRATIFICATION TO PRESERVE CLASS DISTRIBUTION
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,       # 20% DATA RESERVED FOR TESTING
    random_state=42,     # FIXED SEED FOR REPRODUCIBILITY
    stratify=y           # MAINTAIN CLASS RATIO IN BOTH SPLITS
)
print("TRAINING SAMPLES:", X_train.shape)
print("TESTING  SAMPLES:", X_test.shape)
 
# COMPUTE CLASS WEIGHTS TO HANDLE CLASS IMBALANCE DURING TRAINING
print("COMPUTING CLASS WEIGHTS...")
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("CLASS WEIGHTS:", class_weights)
 
# DEFINE THE MODEL INPUT LAYER WITH SHAPE (MFCC BINS, TIME STEPS, CHANNEL)
print("BUILDING CNN + GRU MODEL...")
inputs = tf.keras.Input(shape=(13, 120, 1))
 
# --- FIRST CONVOLUTIONAL BLOCK ---
# EXTRACT LOW-LEVEL SPATIAL FEATURES WITH 32 FILTERS OF SIZE 3x3
x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
# NORMALIZE ACTIVATIONS AFTER CONVOLUTION TO STABILIZE TRAINING
x = tf.keras.layers.BatchNormalization()(x)
# DOWNSAMPLE FEATURE MAPS TO REDUCE SPATIAL DIMENSIONS
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
 
# --- SECOND CONVOLUTIONAL BLOCK ---
# EXTRACT HIGHER-LEVEL FEATURES WITH 64 FILTERS FOR RICHER REPRESENTATIONS
x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
# NORMALIZE ACTIVATIONS AGAIN BEFORE POOLING
x = tf.keras.layers.BatchNormalization()(x)
# FURTHER REDUCE SPATIAL DIMENSIONS BEFORE FEEDING INTO GRU
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
 
# --- TRANSITION TO SEQUENTIAL (TEMPORAL) PROCESSING ---
# RESHAPE 2D FEATURE MAPS INTO SEQUENCES COMPATIBLE WITH THE GRU LAYER
shape = x.shape
x = tf.keras.layers.Reshape((shape[1], shape[2] * shape[3]))(x)
 
# CAPTURE TEMPORAL DEPENDENCIES AND SEQUENTIAL PATTERNS IN THE FEATURE SEQUENCES
x = tf.keras.layers.GRU(64)(x)
 
# --- CLASSIFICATION HEAD ---
# FULLY CONNECTED LAYER FOR HIGH-LEVEL FEATURE COMBINATION
x = tf.keras.layers.Dense(64, activation="relu")(x)
 
# APPLY DROPOUT TO PREVENT OVERFITTING DURING TRAINING
x = tf.keras.layers.Dropout(0.5)(x)
 
# FINAL OUTPUT LAYER — SIGMOID ACTIVATION FOR BINARY CLASSIFICATION (COVID / NON-COVID)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
 
# ASSEMBLE THE FUNCTIONAL API MODEL FROM INPUTS TO OUTPUTS
model = tf.keras.Model(inputs, outputs)
 
# CONFIGURE OPTIMIZER, LOSS FUNCTION, AND EVALUATION METRIC
model.compile(
    optimizer="adam",               # ADAPTIVE LEARNING RATE OPTIMIZER
    loss="binary_crossentropy",     # STANDARD LOSS FOR BINARY CLASSIFICATION
    metrics=["accuracy"]            # TRACK ACCURACY DURING TRAINING
)
 
# PRINT A SUMMARY OF THE MODEL ARCHITECTURE
model.summary()
 
# CONFIGURE EARLY STOPPING TO HALT TRAINING WHEN VALIDATION LOSS STOPS IMPROVING
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",         # WATCH VALIDATION LOSS FOR IMPROVEMENT
    patience=5,                 # STOP AFTER 5 EPOCHS WITH NO IMPROVEMENT
    restore_best_weights=True   # REVERT TO THE BEST CHECKPOINT AFTER STOPPING
)
 
# FIT THE MODEL ON TRAINING DATA WITH CLASS WEIGHTS AND EARLY STOPPING
print("STARTING MODEL TRAINING...")
history = model.fit(
    X_train,
    y_train,
    epochs=50,                      # MAXIMUM NUMBER OF TRAINING EPOCHS
    batch_size=32,                  # NUMBER OF SAMPLES PER GRADIENT UPDATE
    validation_split=0.1,           # USE 10% OF TRAINING DATA FOR VALIDATION
    class_weight=class_weights,     # APPLY CLASS WEIGHTS TO COUNTER IMBALANCE
    callbacks=[early_stop]          # APPLY EARLY STOPPING CALLBACK
)
 
# COMPUTE FINAL LOSS AND ACCURACY ON THE HELD-OUT TEST SET
print("EVALUATING MODEL ON TEST DATA...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"TEST ACCURACY : {acc * 100:.2f}%")
print(f"TEST LOSS     : {loss:.4f}")
 
# SAVE THE FULL MODEL (ARCHITECTURE + WEIGHTS + OPTIMIZER STATE) TO DISK
model.save("covid_cough_model.h5")
print("MODEL SUCCESSFULLY SAVED AS 'covid_cough_model.h5'")
