"""
emotion_module/02_train_emotion_model.py
Simple compatible model — trains locally in ~20-30 mins on CPU.
"""

import os, sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, MODEL_DIR, EMOTION_MODEL_H5, NUM_CLASSES, BATCH_SIZE

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATA_PATH = os.path.join(DATA_DIR, "fer2013_processed.npz")

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Not found: {DATA_PATH}")
    sys.exit(1)

# ── Load ───────────────────────────────────────────────
data    = np.load(DATA_PATH)
X_train = data["X_train"]
y_train = data["y_train"]
X_test  = data["X_test"]
y_test  = data["y_test"]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── Simple Sequential model (fully compatible) ─────────
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ── Callbacks ──────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

cb_list = [
    callbacks.ModelCheckpoint(
        EMOTION_MODEL_H5,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
]

# ── Train ──────────────────────────────────────────────
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=cb_list,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(f"Saved to: {EMOTION_MODEL_H5}")