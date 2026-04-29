from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load model
model = load_model("ethnicity.h5")

ethnicity = ["White", "Black", "Asian", "Indian", "Others"]


@app.route('/')
def home():
    return "Ethnicity Detection API Running 🚀"


@app.route('/predict/<path:id>', methods=['GET'])
def predict(id):
    img_path = f"{id}"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))

    img = img.reshape(1, 48, 48, 1)
    img = img.astype("float32")

    ans = model.predict(img)

    predicted_class = np.argmax(ans)
   
    return jsonify({
        "Ethnicity": ethnicity[predicted_class]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)