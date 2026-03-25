# ILLNESS DETECTION MODEL USING CNN + GRU
# DEEP LEARNING PIPELINE FOR BINARY CLASSIFICATION
 
# IMPORT ALL NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 
# LOAD TRAINING FEATURES AND LABELS FROM NUMPY BINARY FILES
x = np.load("x_final_training.npy")
y = np.load("y_final_training.npy")
 
# LOAD TEST FEATURES AND LABELS FROM NUMPY BINARY FILES
x_test = np.load("x_final_test.npy")
y_test  = np.load("y_final_test.npy")
 
# ADD A CHANNEL DIMENSION TO MAKE DATA COMPATIBLE WITH CONV2D (HEIGHT, WIDTH, CHANNELS)
X = x[..., np.newaxis]
 
# PRINT DATA SHAPES FOR VERIFICATION
print(f"TRAINING FEATURES SHAPE : {x.shape}")
print(f"TRAINING LABELS SHAPE   : {y.shape}")
 
# ASSIGN TRAINING AND TEST SETS
X_train = X
X_test  = x_test
y_train = y
# NOTE: y_test IS ALREADY LOADED ABOVE — KEEPING AS IS
 
# EXTRACT INPUT SHAPE FROM TRAINING DATA (USED TO DEFINE MODEL INPUT LAYER)
input_shape = X_train.shape[1:]
 
# BUILD THE HYBRID CNN + GRU SEQUENTIAL MODEL
model = tf.keras.models.Sequential(name="Illness_Detection_CNN_GRU")
 
# --- FIRST CONVOLUTIONAL BLOCK ---
# EXTRACT LOW-LEVEL SPATIAL FEATURES WITH 32 FILTERS OF SIZE 3x3
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# DOWNSAMPLE FEATURE MAPS TO REDUCE SPATIAL DIMENSIONS
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
 
# --- SECOND CONVOLUTIONAL BLOCK ---
# EXTRACT HIGHER-LEVEL FEATURES WITH 64 FILTERS FOR RICHER REPRESENTATIONS
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# FURTHER DOWNSAMPLE TO REDUCE COMPUTATIONAL COMPLEXITY
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
 
# NORMALIZE ACTIVATIONS TO STABILIZE AND SPEED UP TRAINING
model.add(tf.keras.layers.BatchNormalization())
 
# --- TRANSITION TO SEQUENTIAL (TEMPORAL) PROCESSING ---
# RESHAPE SPATIAL FEATURE MAPS INTO SEQUENCES FOR THE GRU LAYER
model.add(tf.keras.layers.Reshape((-1, 64)))
 
# CAPTURE TEMPORAL/SEQUENTIAL PATTERNS ACROSS THE RESHAPED FEATURE SEQUENCES
model.add(tf.keras.layers.GRU(64, return_sequences=False))
 
# --- CLASSIFICATION HEAD ---
# FULLY CONNECTED LAYER FOR HIGH-LEVEL FEATURE COMBINATION
model.add(tf.keras.layers.Dense(64, activation='relu'))
 
# APPLY DROPOUT TO PREVENT OVERFITTING DURING TRAINING
model.add(tf.keras.layers.Dropout(0.3))
 
# FINAL OUTPUT LAYER — SIGMOID ACTIVATION FOR BINARY CLASSIFICATION (0 OR 1)
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
 
# CONFIGURE OPTIMIZER, LOSS FUNCTION, AND EVALUATION METRIC
model.compile(
    optimizer='adam',                  # ADAPTIVE LEARNING RATE OPTIMIZER
    loss='binary_crossentropy',        # STANDARD LOSS FOR BINARY CLASSIFICATION
    metrics=['accuracy']               # TRACK ACCURACY DURING TRAINING
)
 
# PRINT A SUMMARY OF THE MODEL ARCHITECTURE
model.summary()
 
# FIT THE MODEL ON TRAINING DATA WITH VALIDATION MONITORING
print("\nSTARTING MODEL TRAINING...")
history = model.fit(
    X_train, y_train,
    epochs=35,                          # NUMBER OF COMPLETE PASSES THROUGH TRAINING DATA
    batch_size=32,                      # NUMBER OF SAMPLES PER GRADIENT UPDATE
    validation_data=(X_test, y_test)    # EVALUATE ON TEST DATA AFTER EACH EPOCH
)
 
# COMPUTE FINAL LOSS AND ACCURACY ON THE TEST SET
loss, acc = model.evaluate(X_test, y_test, verbose=0)
 
print(f"\nTEST ACCURACY : {acc * 100:.2f}%")
print(f"TEST LOSS     : {loss:.4f}")
 
# VISUALIZE TRAINING VS VALIDATION ACCURACY OVER EPOCHS
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'],     label='TRAIN ACCURACY',      linewidth=2)
plt.plot(history.history['val_accuracy'], label='VALIDATION ACCURACY', linewidth=2, linestyle='--')
plt.title('MODEL ACCURACY OVER EPOCHS', fontsize=14, fontweight='bold')
plt.xlabel('EPOCH', fontsize=12)
plt.ylabel('ACCURACY', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
 
# SAVE THE FULL MODEL (ARCHITECTURE + WEIGHTS + OPTIMIZER STATE) TO DISK
model.save("illness_detection_model.h5")
print("MODEL SUCCESSFULLY SAVED AS 'illness_detection_model.h5'")
