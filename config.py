import os


USE_EMOTION    = True
USE_ETHNICITY  = False   # ← now enabled; set False if model not downloaded yet
USE_DECEPTION  = True


BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR           = os.path.join(BASE_DIR, "models")

EMOTION_MODEL_PATH   = os.path.join(MODELS_DIR, "emotion_model.h5")
EMOTION_MODEL_H5     = os.path.join(MODELS_DIR, "emotion_model.h5")     # ← added
EMOTION_MODEL_KERAS  = os.path.join(MODELS_DIR, "emotion_model.keras")  # ← added
ETHNICITY_MODEL_PATH = os.path.join(BASE_DIR, "ethnicity_module", "ethnicity.h5")
ETHNICITY_API_URL = "http://127.0.0.1:3000/predict"

MEDIAPIPE_MODEL_SELECTION = 0
MEDIAPIPE_MIN_CONFIDENCE  = 0.6   # raise to 0.75 if you get false positives

IMG_SIZE = (48, 48)     # (width, height) — same for both models

CONFIDENCE_THRESHOLD = 0.4       # ← added (adjust if predictions feel too uncertain)

# ── Labels ────────────────────────────────────────────────────────────────────
EMOTION_LABELS   = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
ETHNICITY_LABELS = ["White", "Black", "Asian", "Indian", "Others"]

# ── Deception thresholds ──────────────────────────────────────────────────────
DECEPTION_THRESHOLD        = 0.6    # score above this → "Deceptive"
MICROEXPRESSION_WINDOW_SEC = 0.5    # seconds to track rapid expression changes
SUSPICIOUS_EMOTIONS        = {"Fear", "Disgust"}  # emotions that raise score

# ── Visualization ─────────────────────────────────────────────────────────────
DASHBOARD_FONT_SCALE  = 0.55
DASHBOARD_THICKNESS   = 1
FACE_BOX_COLOR        = (0, 255, 0)     # BGR green
EMOTION_TEXT_COLOR    = (255, 255, 255) # white
ETHNICITY_TEXT_COLOR  = (0, 200, 255)   # yellow-ish
DECEPTION_TRUE_COLOR  = (0, 0, 255)     # red
DECEPTION_FALSE_COLOR = (0, 255, 0)     # green