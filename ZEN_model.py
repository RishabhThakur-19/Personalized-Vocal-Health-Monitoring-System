import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# ===============================
# LOAD DATA
# ===============================

print("Loading dataset...")

X = np.load("X_new.npy")
y = np.load("y_new.npy")

print("Original dataset shape:", X.shape)

# ===============================
# NORMALIZATION
# ===============================

print("Normalizing data...")

X = (X - np.mean(X)) / np.std(X)

# Add CNN channel dimension
X = X[..., np.newaxis]

print("New dataset shape:", X.shape)

# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# ===============================
# CLASS BALANCING
# ===============================

print("Computing class weights...")

class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

# ===============================
# BUILD MODEL
# ===============================

print("Building CNN + GRU model...")

inputs = tf.keras.Input(shape=(13, 120, 1))

# CNN block 1
x = tf.keras.layers.Conv2D(32, (3,3), activation="relu")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

# CNN block 2
x = tf.keras.layers.Conv2D(64, (3,3), activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

# reshape for GRU
shape = x.shape
x = tf.keras.layers.Reshape((shape[1], shape[2]*shape[3]))(x)

# GRU layer
x = tf.keras.layers.GRU(64)(x)

# Dense layers
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

# ===============================
# COMPILE MODEL
# ===============================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# EARLY STOPPING
# ===============================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ===============================
# TRAIN MODEL
# ===============================

print("Starting training...")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# ===============================
# EVALUATE MODEL
# ===============================

print("Evaluating model...")

loss, acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", acc)

# ===============================
# SAVE MODEL
# ===============================

model.save("covid_cough_model.h5")

print("Model saved as covid_cough_model.h5")