# Emotion Deception Project

A real-time facial analysis pipeline that detects **emotion**, **ethnicity**, and **deception** from a live webcam feed.

## Project Structure

```
Emotion_Deception_Project/
├── config.py                      # Central config — paths, labels, thresholds
├── main.py                        # Entry point — orchestrates all modules
├── face_module/
│   └── face_detector.py           # Haar cascade face detection
├── emotion_module/
│   ├── emotion_inference.py       # Loads model, predicts emotion
│   └── 02_train_emotion_model.py  # Local training script
├── deception_module/
│   └── deception_logic.py         # Rule-based deception scoring
├── ethnicity_module/
│   └── ethnicity_model.py         # Ethnicity prediction (in progress)
├── visualization/
│   └── dashboard.py               # Real-time dashboard window
└── notebooks/
    ├── 01_emotion_dataset_preprocessing.ipynb
    └── 02_emotion_model_training.ipynb
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/HarshSahu550/Emotion_Deception_Project.git
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

## Modules

| Module | Status |
|--------|--------|
| emotion_module | Done |
| face_module | Done |
| deception_module | Done |
| ethnicity_module | In Progress |
| visualization | Done |

## Dataset

- **FER2013** — 35,887 grayscale 48x48 face images, 7 emotion classes
- Sourced from Kaggle: `msambare/fer2013`

## Model

- Architecture: CNN with BatchNormalization + GlobalAveragePooling
- Input: 48x48 grayscale
- Output: 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- Test accuracy: ~59% on FER2013

## Notes

- Model file is NOT in the repo — use the gdown command above to download it
- `data/` folder is not needed unless you want to retrain
- Set `USE_ETHNICITY = True` in `main.py` once ethnicity module is ready
