# ethnicity_module/ethnicity_model.py
"""
Calls the Flask ethnicity API with a face crop (BGR numpy array).
Returns (label, confidence) or ("Unknown", 0.0) on failure.
"""

import cv2
import numpy as np
import base64
import requests

from config import ETHNICITY_API_URL

ETHNICITY_LABELS = ["White", "Black", "Asian", "Indian", "Others"]


def predict_ethnicity(face_bgr: np.ndarray) -> tuple[str, float]:
    """
    Takes a BGR face crop (any size), sends it to the Flask API,
    returns (ethnicity_label, confidence).
    Falls back to ("Unknown", 0.0) if API is unreachable.
    """
    if face_bgr is None or face_bgr.size == 0:
        return "Unknown", 0.0

    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))

        _, buf = cv2.imencode(".png", gray)
        b64 = base64.b64encode(buf).decode("utf-8")

        resp = requests.post(
            ETHNICITY_API_URL,
            json={"image": b64},
            timeout=0.5          # 500ms — fast enough for real-time
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("ethnicity", "Unknown"), data.get("confidence", 0.0)

    except Exception:
        return "Unknown", 0.0

class EthnicityPredictor:
    def is_ready(self):
        return True  # always ready — delegates to Flask API

    def predict(self, face_bgr: np.ndarray) -> dict:
        label, confidence = predict_ethnicity(face_bgr)
        return {
            "label":      label,
            "confidence": confidence,
            "scores":     {}   # API doesn't return per-class scores
        }