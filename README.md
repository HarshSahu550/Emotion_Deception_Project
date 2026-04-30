# Emotion Deception Project
A real-time facial analysis pipeline that detects emotion, ethnicity, and deception from a live webcam feed.

## Project Structure
```
Emotion_Deception_Project/
├── config.py                         # Central config — paths, labels, API endpoints
├── main.py                           # Entry point — orchestrates all modules
├── app.py                            # Flask REST API — serves ethnicity model
├── test_pipeline.py                  # All pipeline tests (run before main.py)
├── requirements.txt
├── detector.tflite                   # MediaPipe BlazeFace model (download separately)
├── face_module/
│   ├── face_detector.py              # MediaPipe TFLite face detection
│   └── __init__.py
├── emotion_module/
│   ├── emotion_inference.py          # Loads model, runs emotion prediction
│   ├── 02_train_emotion_model.py     # Training script
│   └── __init__.py
├── deception_module/
│   ├── deception_logic.py            # Rule-based deception scoring
│   └── __init__.py
├── ethnicity_module/
│   ├── app.py                        # Flask REST API — serves ethnicity.h5
│   ├── ethnicity_model.py            # Bridge: calls Flask API from pipeline
│   ├── ethnicity.h5                  # Ethnicity model (not in git)
│   └── __init__.py
├── visualization/
│   ├── dashboard.py                  # Real-time OpenCV dashboard window
│   └── __init__.py
├── models/                           # Not in git — download separately
│   ├── emotion_model.h5
│   └── emotion_model.keras
└── notebooks/
    ├── 01_emotion_dataset_preprocessing.ipynb
    └── 02_emotion_model_training.ipynb
```

## Setup

### 1. Clone
```bash
git clone https://github.com/HarshSahu550/Emotion_Deception_Project.git
cd Emotion_Deception_Project
```

### 2. Virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 3. Download model files
```bash
pip install gdown

# Emotion model
gdown https://drive.google.com/uc?id=1mM4I3sVYyw-MM-L6-I72Nyo_RFuGxIlM -O models/emotion_model.h5

# Ethnicity model — place at:
# ethnicity_module/ethnicity.h5
```

### 4. Download MediaPipe face detector model
```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" -OutFile "detector.tflite"
```
```bash
# Mac/Linux
wget -O detector.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```

### 5. Start the ethnicity Flask API (Terminal 1)
```bash
python app.py
# Runs on http://localhost:3000
```

### 6. Run tests (Terminal 2)
```bash
python test_pipeline.py
```

### 7. Run the pipeline (Terminal 2)
```bash
python main.py
```
Press `Q` to quit.

---

## How it works
Each video frame goes through this pipeline:

```
Webcam → Face Detection → Emotion CNN → Deception Scorer → Display
                       ↘ Ethnicity API ↗
```

- `face_module` detects faces using **MediaPipe BlazeFace** (TFLite)
- `emotion_module` runs a CNN (trained on FER2013) to get 7 emotion probabilities
- `ethnicity_module` sends the face to a Flask API which runs a second CNN
- `deception_module` scores deception likelihood from the rolling emotion history
- `visualization` renders everything on the webcam feed and a separate dashboard

---

## Modules & Credits

| Module | Who | Roll No |
|---|---|---|
| Emotion dataset preprocessing | Harsh Sahu | 23051269 |
| Emotion model training | Harsh Sahu | 23051269 |
| `emotion_module/emotion_inference.py` | Harsh Sahu | 23051269 |
| `face_module/face_detector.py` (MediaPipe migration) | Harsh Sahu | 23051269 |
| `main.py` (full integration) | Harsh Sahu | 23051269 |
| `config.py` (all updates & fixes) | Harsh Sahu | 23051269 |
| `deception_module/deception_logic.py` | Vijeta | 23051314 |
| `visualization/dashboard.py` | Vijeta + Harsh | 23051314, 23051269 |
| Ethnicity model training | Sachin Kumar Arya | 23051291 |
| `ethnicity_module/ethnicity_model.py` | Sachin Kumar Arya | 23051291 |
| `ethnicity_module/app.py` (Flask API) | Md. Kamil | 23053344 |
| Pipeline tests (emotion + deception) | Aadhya Thakre | 23053180 |
| Ethnicity model tests | Tanya Singh | 23051310 |

---

## Configuration (`config.py`)

| Key | Default | Description |
|---|---|---|
| `USE_ETHNICITY` | `True` | Enable ethnicity module (needs Flask API) |
| `ETHNICITY_API_URL` | `http://localhost:3000/predict` | Flask endpoint |
| `EMOTION_MODEL_H5` | `models/emotion_model.h5` | Emotion model path |
| `EMOTION_MODEL_KERAS` | `models/emotion_model.keras` | Emotion model (Keras format) |
| `CONFIDENCE_THRESHOLD` | `0.4` | Min confidence to show emotion label |

---

## Models

| Model | Dataset | Architecture | Input | Output | Accuracy |
|---|---|---|---|---|---|
| `emotion_model.h5` | FER2013 (35,887 images) | 3× Conv + GAP | 48×48 grayscale | 7 emotions | ~59% |
| `ethnicity.h5` | Age-Gender CSV | 3× Conv + GAP | 48×48 grayscale | 5 ethnicities | — |

Both use: Conv2D → BatchNorm → MaxPool → Dropout, GlobalAveragePooling, Dense(256), Adam(lr=0.0003), EarlyStopping(patience=7).

---

## Ethnicity API

```
POST http://localhost:3000/predict
Body: {"image": "<base64-encoded PNG>"}

Response: {"ethnicity": "Indian", "confidence": 0.87}
```

---

## Git Workflow (for contributors)

```bash
# Always start from a fresh main
git checkout main
git pull origin main
git checkout -b your-branch-name

# Make your changes, then:
git add <only your files>
git commit -m "type(scope): description"
git push origin your-branch-name
# Open a Pull Request — don't merge your own PR
```

**Rules:**
- Never push directly to `main`
- Only touch your own module folder
- Always pull before starting work
- Open a PR and get one approval before merging

---

## Notes

- `models/` and `ethnicity_module/ethnicity.h5` are gitignored — never commit `.h5` or `.keras` files (too large)
- `detector.tflite` must be downloaded separately and placed in the project root
- If ethnicity API is down, the pipeline degrades gracefully — shows `"Unknown"` and keeps running
- Set `USE_ETHNICITY = False` in `config.py` to run without the Flask API
- GPU is not supported on native Windows with TensorFlow ≥ 2.11 — use WSL2 for GPU acceleration