# Emotion Deception Project

A real-time facial analysis pipeline that detects **emotion**, **ethnicity**, and **deception** from a live webcam feed.

## Project Structure

```
Emotion_Deception_Project/
├── config.py                      # Central config — paths, labels, thresholds
├── main.py                        # Entry point — orchestrates all modules
├── test_pipeline.py               # Unit + integration tests
├── requirements.txt               # Python dependencies
├── face_module/
│   ├── face_detector.py           # Haar cascade face detection
│   └── __init__.py
├── emotion_module/
│   ├── emotion_inference.py       # Loads model, predicts emotion
│   ├── 02_train_emotion_model.py  # Local training script
│   └── __init__.py
├── deception_module/
│   ├── deception_logic.py         # Rule-based deception scoring
│   └── __init__.py
├── ethnicity_module/
│   ├── ethnicity_model.py         # Ethnicity prediction (in progress)
│   └── __init__.py
├── visualization/
│   ├── dashboard.py               # Real-time OpenCV dashboard
│   └── __init__.py
└── notebooks/
    ├── 01_emotion_dataset_preprocessing.ipynb
    └── 02_emotion_model_training.ipynb
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Team01Sahu550/Emotion_Deception_Project.git
cd Emotion_Deception_Project
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the model file
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1mM4I3sVYyw-MM-L6-I72Nyo_RFuGxIlM -O models/emotion_model.h5
```

### 4. Run
```bash
python main.py
```
Press `Q` to quit.

### 5. Run tests
```bash
python test_pipeline.py
```
All 22 tests should pass before running the main pipeline.

## Modules

| Module | Owner | Status |
|--------|-------|--------|
| emotion_module | Team01 | Done |
| face_module | Team01 | Done |
| deception_module | Team01 | Done |
| ethnicity_module | Team02 | In Progress |
| visualization | Team01 | Done |

## Dataset

- **FER2013** — 35,887 grayscale 48x48 face images, 7 emotion classes
- Sourced from Kaggle: `msambare/fer2013`
- Preprocessing: `notebooks/01_emotion_dataset_preprocessing.ipynb`
- Training: `notebooks/02_emotion_model_training.ipynb`

## Model

- Architecture: CNN (3 Conv blocks + GlobalAveragePooling)
- Input: 48x48 grayscale
- Output: 7 emotions — Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Test accuracy: ~59% on FER2013
- Format: `.h5` (TensorFlow 2.15 compatible)

## Notes

- Model file (`.h5`) and data file (`.npz`) are excluded from git — too large
- Get model via `gdown` command above
- Data file only needed for retraining — not required to run
- Set `USE_ETHNICITY = True` in `main.py` once ethnicity module is ready
- Set `USE_DECEPTION = True` in `main.py` once deception module is ready (currently on)

## Git Workflow for Contributors

### Every time you work
```bash
git pull                        # always pull first
# ... make your changes ...
git checkout -b your-branch-name
git add .
git commit -m "describe what you changed"
git push origin your-branch-name
# then open a Pull Request on GitHub
```

### Rules
- Never push directly to `main`
- Only touch your own module folder
- Always pull before starting work
- Open a Pull Request and wait for approval before merging