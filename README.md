# Emotion Deception Project

A real-time facial analysis pipeline that detects **emotion**, **ethnicity**, and **deception** from a live webcam feed.

## Project Structure

```
Emotion_Deception_Project/
├── config.py                         # Central config — paths, labels, API endpoints
├── main.py                           # Entry point — orchestrates all modules
├── test_pipeline.py                  # All pipeline tests (run before main.py)
├── requirements.txt
├── face_module/
│   ├── face_detector.py              # Haar cascade face detection
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
│   └── __init__.py
├── visualization/
│   ├── dashboard.py                  # Real-time OpenCV dashboard window
│   └── __init__.py
├── models/                           # Not in git — download separately
│   ├── emotion_model.h5
│   └── ethnicity.h5
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

# Ethnicity model — get link from team and place at:
# models/ethnicity.h5
```

### 4. Start the ethnicity Flask API (Terminal 1)
```bash
python ethnicity_module/app.py
# Runs on http://localhost:3000
```

### 5. Run tests (Terminal 2)
```bash
python test_pipeline.py
```

### 6. Run the pipeline (Terminal 2)
```bash
python main.py
```
Press `Q` to quit.

## How it works

Each video frame goes through this pipeline:

```
Webcam → Face Detection → Emotion CNN → Deception Scorer → Display
                       ↘ Ethnicity API ↗
```

1. **face_module** detects faces using Haar cascades
2. **emotion_module** runs a CNN (trained on FER2013) to get 7 emotion probabilities
3. **ethnicity_module** sends the face to a Flask API which runs a second CNN
4. **deception_module** scores deception likelihood from the rolling emotion history
5. **visualization** renders everything on the webcam feed and a separate dashboard

## Modules & Credits

| Module | Who | Roll No |
|--------|-----|---------|
| Emotion dataset preprocessing | Harsh Sahu | 23051269 |
| Emotion model training | Harsh Sahu | 23051269 |
| `emotion_module/emotion_inference.py` | Harsh Sahu | 23051269 |
| `face_module/face_detector.py` | Harsh Sahu | 23051269 |
| `main.py` (full integration) | Harsh Sahu | 23051269 |
| `deception_module/deception_logic.py` | Vijeta | 23051314 |
| `visualization/dashboard.py` | Vijeta + Harsh | 23051314, 23051269 |
| Ethnicity model training | Sachin Kumar Arya | 23051291 |
| `ethnicity_module/ethnicity_model.py` | Sachin Kumar Arya | 23051291 |
| `ethnicity_module/app.py` (Flask API) | Md. Kamil | 23053344 |
| Pipeline tests (emotion + deception) | Aadhya Thakre | 23053180 |
| Ethnicity model tests | Tanya Singh | 23051310 |

## Configuration (`config.py`)

| Key | Default | Description |
|-----|---------|-------------|
| `USE_ETHNICITY` | `True` | Enable ethnicity module (needs Flask API) |
| `ETHNICITY_API_URL` | `http://localhost:3000/predict` | Flask endpoint |
| `EMOTION_MODEL_H5` | `models/emotion_model.h5` | Emotion model path |
| `CONFIDENCE_THRESHOLD` | `0.25` | Min confidence to show emotion label |

## Models

| Model | Dataset | Architecture | Input | Output | Accuracy |
|-------|---------|-------------|-------|--------|----------|
| `emotion_model.h5` | FER2013 (35,887 images) | 3× Conv + GAP | 48×48 grayscale | 7 emotions | ~59% |
| `ethnicity.h5` | Age-Gender CSV | 3× Conv + GAP | 48×48 grayscale | 5 ethnicities | — |

Both use: Conv2D → BatchNorm → MaxPool → Dropout, GlobalAveragePooling, Dense(256), Adam(lr=0.0003), EarlyStopping(patience=7).

## Ethnicity API

```
POST http://localhost:3000/predict
Body: {"image": "<base64-encoded PNG>"}

Response: {"ethnicity": "Indian", "confidence": 0.87}
```

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

## Notes

- `models/` is gitignored — never commit `.h5` files (too large)
- If ethnicity API is down, the pipeline degrades gracefully — shows "Unknown" and keeps running
- Set `USE_ETHNICITY = False` in `main.py` to run without the Flask API