

import cv2
import sys

from config import (
    USE_EMOTION, USE_ETHNICITY, USE_DECEPTION,
    MEDIAPIPE_MODEL_SELECTION, MEDIAPIPE_MIN_CONFIDENCE
)
from face_module.face_detector       import FaceDetector
from emotion_module.emotion_inference   import EmotionPredictor
if USE_ETHNICITY:
    from ethnicity_module.ethnicity_model import EthnicityPredictor
from deception_module.deception_logic   import DeceptionScorer
from visualization.dashboard            import Dashboard


def main():
    # ── Initialise modules ────────────────────────────────────────────────
    print("[main] Initialising pipeline...")

    face_detector = FaceDetector(
        min_confidence=MEDIAPIPE_MIN_CONFIDENCE,
        model_selection=MEDIAPIPE_MODEL_SELECTION
    )
    dashboard     = Dashboard()

    emotion_predictor   = EmotionPredictor()   if USE_EMOTION    else None
    ethnicity_predictor = EthnicityPredictor() if USE_ETHNICITY  else None
    deception_scorer    = DeceptionScorer()    if USE_DECEPTION  else None

    # ── Open webcam ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[main] ERROR: Cannot open webcam. Exiting.")
        sys.exit(1)

    print("[main] Pipeline running — press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[main] WARNING: Empty frame received.")
            continue

        # ── Detect faces ──────────────────────────────────────────────────
        faces = face_detector.detect(frame)

        results = []

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            result = {"bbox": (x, y, w, h)}

            # ── Emotion ───────────────────────────────────────────────────
            if emotion_predictor and emotion_predictor.is_ready():
                emotion_result = emotion_predictor.predict(face_roi)
                result["emotion"] = emotion_result
            else:
                result["emotion"] = {"label": "N/A", "confidence": 0.0}

            # ── Ethnicity ─────────────────────────────────────────────────
            if ethnicity_predictor and ethnicity_predictor.is_ready():
                ethnicity_result = ethnicity_predictor.predict(face_roi)
                result["ethnicity"] = ethnicity_result
            else:
                result["ethnicity"] = {"label": "N/A", "confidence": 0.0}

            # ── Deception ─────────────────────────────────────────────────
            if deception_scorer:
                deception_result = deception_scorer.score(result["emotion"])
                result["deception"] = deception_result
            else:
                result["deception"] = {"label": "N/A", "score": 0.0}

            results.append(result)

        # ── Render dashboard ──────────────────────────────────────────────
        display_frame = dashboard.render(frame, results)
        cv2.imshow("Emotion | Ethnicity | Deception", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[main] Quit signal received.")
            break

    cap.release()
    face_detector.close()
    cv2.destroyAllWindows()
    print("[main] Pipeline stopped.")


if __name__ == "__main__":
    main()