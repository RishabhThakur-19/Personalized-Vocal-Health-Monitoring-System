import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

# ─── 1. Load data ─────────────────────────────────────────────
x = np.load("x_final_training_v2.npy")
y = np.load("y_final_training_v2.npy")
x_test = np.load("x_final_test_v2.npy")
y_test = np.load("y_final_test_v2.npy")

print("Train X:", x.shape, "Train y:", y.shape)
print("Test  X:", x_test.shape, "Test  y:", y_test.shape)

# ─── 2. Normalisation (CRITICAL) ──────────────────────────────
n_samples, n_frames, n_features = x.shape

scaler = StandardScaler()

x_flat = x.reshape(-1, n_features)
x = scaler.fit_transform(x_flat).reshape(n_samples, n_frames, n_features)

x_test_flat = x_test.reshape(-1, n_features)
x_test = scaler.transform(x_test_flat).reshape(x_test.shape)

# ─── 3. Add channel axis ──────────────────────────────────────
X = x[..., np.newaxis]
X_test = x_test[..., np.newaxis]

# ─── 4. Train / Validation split ──────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

input_shape = X_train.shape[1:]

# ─── 5. Model ─────────────────────────────────────────────────
model = tf.keras.models.Sequential()

# CNN feature extractor
model.add(tf.keras.layers.Conv2D(
    32, (3,3), activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    input_shape=input_shape
))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Conv2D(
    64, (3,3), activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.BatchNormalization())

# ─── KEY FIX: Proper reshape for GRU ──────────────────────────
# We treat "frames" as time and collapse spatial feature maps
shape = model.output_shape   # (None, H, W, C)

time_steps = shape[1]        # reduced frames after pooling
features   = shape[2] * shape[3]

model.add(tf.keras.layers.Reshape((time_steps, features)))

# GRU (now actually meaningful)
model.add(tf.keras.layers.GRU(
    64,
    return_sequences=False,
    dropout=0.2,
    recurrent_dropout=0.1
))

# Classifier head
model.add(tf.keras.layers.Dense(
    64, activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# ─── 6. Compile ───────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# ─── 7. Callbacks ─────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
]

# ─── 8. Train ────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# ─── 9. Evaluate ─────────────────────────────────────────────
loss, acc, prec, rec = model.evaluate(X_test, y_test)

print("\nTest Results:")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("Loss     :", loss)

# ─── 10. Plot ────────────────────────────────────────────────
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.legend()

plt.show()

# ─── 11. Save ────────────────────────────────────────────────
model.save("illness_detection_model_gru_v1.h5")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model + scaler saved successfully")