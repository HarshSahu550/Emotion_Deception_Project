# Emotion Deception Project

A real-time facial analysis pipeline that detects **emotion**, **ethnicity**, and **deception** from a live webcam feed.

## Project Structure

```
Emotion_Deception_Project/
├── config.py                  # Central config — paths, labels, thresholds
├── main.py                    # Entry point — orchestrates all modules
├── face_module/
│   └── face_detector.py       # Haar cascade face detection
├── emotion_module/
│   ├── emotion_inference.py   # Loads model, predicts emotion
│   └── 02_train_emotion_model.py  # Local training script
├── deception_module/
│   └── deception_logic.py     # Rule-based deception scoring
├── ethnicity_module/
│   └── ethnicity_model.py     # Ethnicity prediction (in progress)
├── visualization/
│   └── dashboard.py           # Real-time dashboard window
└── notebooks/
    ├── 01_emotion_dataset_preprocessing.ipynb
    └── 02_emotion_model_training.ipynb
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Emotion_Deception_Project.git
cd Emotion_Deception_Project
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download model and data files
Model and data files are NOT in this repo (too large).
Download from the shared Google Drive folder and place them as:
```
models/emotion_model.h5
data/fer2013_processed.npz
```

### 5. Run
```bash
python main.py
```
Press `Q` to quit.

## Modules

| Module | Owner | Status |
|--------|-------|--------|
| emotion_module | [Team1] | Done |
| face_module | [Team1] | Done |
| deception_module | [Team1] | Done |
| ethnicity_module | [Teammate] | In Progress |
| visualization | [Team1] | Done |

## Dataset

- **FER2013** — 35,887 grayscale 48x48 face images, 7 emotion classes
- Sourced from Kaggle: `msambare/fer2013`
- Preprocessing notebook: `notebooks/01_emotion_dataset_preprocessing.ipynb`

## Model

- Architecture: CNN with BatchNormalization + GlobalAveragePooling
- Input: 48x48 grayscale
- Output: 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- Test accuracy: ~59% on FER2013

## Notes

- Model files (.h5) and data files (.npz) are excluded from git — get them from shared Drive
- Set `USE_ETHNICITY = True` in `main.py` once ethnicity module is ready
- Set `USE_DECEPTION = True` in `main.py` once deception module is ready