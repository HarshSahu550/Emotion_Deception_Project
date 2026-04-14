import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import (
    BOX_COLOR, TEXT_COLOR, FONT_SCALE, FONT_THICK, EMOTION_LABELS
)
from face_module.face_detector import detect_faces
from emotion_module.emotion_inference import predict_emotion, predict_emotion_all
from deception_module.deception_logic import update_and_score, reset
from visualization.dashboard import Dashboard

# ── CHANGED: flip this to True ────────────────────────
USE_ETHNICITY = True

# ── NEW: import ethnicity module ──────────────────────
if USE_ETHNICITY:
    from ethnicity_module.ethnicity_model import predict_ethnicity

BAR_W  = 200
BAR_H  = 160
BAR_PAD = 6
BAR_COLORS = {
    "Angry":    (0,   0,   220),
    "Disgust":  (0,   140, 0  ),
    "Fear":     (130, 0,   130),
    "Happy":    (0,   210, 255),
    "Neutral":  (160, 160, 160),
    "Sad":      (200, 100, 0  ),
    "Surprise": (0,   165, 255),
}
DECEPTION_COLORS = {
    "Low":          (0,   200, 0  ),
    "Medium":       (0,   165, 255),
    "High":         (0,   0,   220),
    "Analyzing...": (160, 160, 160),
}


def draw_label(frame, text, x, y, color=TEXT_COLOR):
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)
    cv2.rectangle(frame, (x, y-th-6), (x+tw+4, y+2), (0,0,0), -1)
    cv2.putText(frame, text, (x+2, y),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                color, FONT_THICK, cv2.LINE_AA)


def draw_emotion_bars(frame, all_probs, origin_x, origin_y):
    panel = frame[origin_y:origin_y+BAR_H, origin_x:origin_x+BAR_W].copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0,0), (BAR_W, BAR_H), (20,20,20), -1)
    panel = cv2.addWeighted(overlay, 0.6, panel, 0.4, 0)

    n       = len(EMOTION_LABELS)
    row_h   = (BAR_H - BAR_PAD*2) // n
    max_bar = BAR_W - 70

    for i, emotion in enumerate(EMOTION_LABELS):
        prob  = all_probs.get(emotion, 0.0)
        y_top = BAR_PAD + i * row_h
        y_mid = y_top + row_h // 2

        cv2.putText(panel, emotion[:3], (2, y_mid+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1, cv2.LINE_AA)

        bar_len = int(prob * max_bar)
        color   = BAR_COLORS.get(emotion, (180,180,180))
        cv2.rectangle(panel, (38, y_top+2), (38+bar_len, y_top+row_h-2),
                      color, -1)

        cv2.putText(panel, f"{prob*100:.0f}%", (38+bar_len+3, y_mid+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220,220,220), 1, cv2.LINE_AA)

    frame[origin_y:origin_y+BAR_H, origin_x:origin_x+BAR_W] = panel


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    dashboard = Dashboard()
    print("Running — press Q to quit.")
    if USE_ETHNICITY:
        print("[INFO] Ethnicity module ON — make sure Flask API is running on port 3000.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces       = detect_faces(frame)
        fh, fw      = frame.shape[:2]
        chart_drawn = False

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]

            emotion, confidence  = predict_emotion(face_crop)
            all_probs            = predict_emotion_all(face_crop)
            dec_score, dec_label = update_and_score(all_probs)

            # ── NEW: ethnicity ────────────────────────
            if USE_ETHNICITY:
                ethnicity, eth_conf = predict_ethnicity(face_crop)
            else:
                ethnicity, eth_conf = None, 0.0

            # face box
            cv2.rectangle(frame, (x,y), (x+w,y+h), BOX_COLOR, 2)

            # emotion label  (y - 58 to make room for two more lines below)
            draw_label(frame, f"{emotion} {confidence:.0%}", x, y - 58)

            # ── NEW: ethnicity label ──────────────────
            if USE_ETHNICITY and ethnicity:
                draw_label(frame, f"Ethnicity: {ethnicity} {eth_conf:.0%}",
                           x, y - 34)

            # deception label
            dec_color = DECEPTION_COLORS.get(dec_label, TEXT_COLOR)
            draw_label(frame, f"Deception: {dec_label}", x, y - 10, dec_color)

            # bar chart (top-right, first face only)
            if not chart_drawn:
                cx = fw - BAR_W - 10
                cy = 10
                draw_emotion_bars(frame, all_probs, cx, cy)
                chart_drawn = True

            # update dashboard
            dashboard.update(all_probs, dec_score, dec_label,
                             emotion, confidence)

        if not faces:
            reset()

        cv2.imshow("Emotion Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    dashboard.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()