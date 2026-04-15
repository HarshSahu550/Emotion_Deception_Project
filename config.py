import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

EMOTION_MODEL_KERAS = os.path.join(MODEL_DIR, "emotion_model.keras")
EMOTION_MODEL_H5    = os.path.join(MODEL_DIR, "emotion_model.h5")

# ── Labels (alphabetical = training order) ─────────────
# angry=0, disgust=1, fear=2, happy=3, neutral=4, sad=5, surprise=6
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ── Model input ────────────────────────────────────────
IMG_SIZE    = (48, 48)
NUM_CLASSES = 7

# ── Training ───────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 50

# ── Face detection ─────────────────────────────────────
FACE_SCALE_FACTOR  = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE      = (30, 30)

# ── Inference ──────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25
# ── Display (BGR for OpenCV) ───────────────────────────
BOX_COLOR  = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
FONT_SCALE = 0.8
FONT_THICK = 2

# ── Ethnicity API ──────────────────────────────────────
ETHNICITY_API_URL  = "http://localhost:3000/predict"
ETHNICITY_LABELS   = ["White", "Black", "Asian", "Indian", "Others"]
