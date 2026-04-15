from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ethnicity.h5")
model = load_model(MODEL_PATH)

ETHNICITY_LABELS = ["White", "Black", "Asian", "Indian", "Others"]


@app.route('/')
def home():
    return "Ethnicity Detection API Running"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON: {"image": "<base64-encoded PNG bytes>"}
    Returns:      {"ethnicity": "Indian", "confidence": 0.87}
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field"}), 400

    try:
        img_bytes = base64.b64decode(data["image"])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = img.reshape(1, 48, 48, 1).astype("float32") / 255.0

        probs = model.predict(img, verbose=0)[0]
        idx = int(np.argmax(probs))

        return jsonify({
            "ethnicity":  ETHNICITY_LABELS[idx],
            "confidence": float(probs[idx])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)