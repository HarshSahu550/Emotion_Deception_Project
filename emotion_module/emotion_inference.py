import cv2
import numpy as np
import tensorflow as tf
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    EMOTION_MODEL_KERAS, EMOTION_MODEL_H5,
    EMOTION_LABELS, IMG_SIZE, CONFIDENCE_THRESHOLD
)

# ── Load model once at import ──────────────────────────
def _load_model():
    if os.path.exists(EMOTION_MODEL_H5):
        print(f"[emotion_module] Loading: {EMOTION_MODEL_H5}")
        return tf.keras.models.load_model(EMOTION_MODEL_H5)
    raise FileNotFoundError(
        "No model found. Check models/emotion_model.h5 exists."
    )

_model = _load_model()

# ── Preprocessing ──────────────────────────────────────
def _preprocess(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

# ── Public API ─────────────────────────────────────────
def predict_emotion(face_bgr):
    """Returns (label, confidence) for the top emotion."""
    if face_bgr is None or face_bgr.size == 0:
        return "No face", 0.0

    probs      = _model.predict(_preprocess(face_bgr), verbose=0)[0]
    idx        = int(np.argmax(probs))
    confidence = float(probs[idx])
    label      = EMOTION_LABELS[idx] if confidence >= CONFIDENCE_THRESHOLD else "Uncertain"

    return label, confidence


def predict_emotion_all(face_bgr):
    if face_bgr is None or face_bgr.size == 0:
        return {label: 0.0 for label in EMOTION_LABELS}

    inp    = _preprocess(face_bgr)
    logits = _model.predict(inp, verbose=0)[0]

    # Temperature scaling — higher T = softer, more spread out probabilities
    T      = 2.0
    scaled = np.exp(np.log(logits + 1e-8) / T)
    probs  = scaled / scaled.sum()

    return {label: float(p) for label, p in zip(EMOTION_LABELS, probs)}

class EmotionPredictor:
    def is_ready(self):
        return _model is not None

    def predict(self, face_bgr):
        label, confidence = predict_emotion(face_bgr)
        all_probs = predict_emotion_all(face_bgr)
        return {"label": label, "confidence": confidence, "scores": all_probs}