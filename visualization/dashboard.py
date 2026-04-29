"""
visualization/dashboard.py

Separate dashboard window showing:
  - Rolling emotion probability lines (last 10 seconds)
  - Deception score gauge
  - Dominant emotion history bar

Run alongside main.py — call update() every frame,
the window refreshes automatically.
"""

import collections
import numpy as np
import cv2

# ── Settings ───────────────────────────────────────────
HISTORY_LEN  = 150   # frames (~5 sec at 30fps)
WIN_W, WIN_H = 600, 500
WIN_NAME     = "Dashboard — Emotion & Deception"

EMOTION_COLORS_BGR = {
    "Angry":    (0,   0,   220),
    "Disgust":  (0,   140, 0  ),
    "Fear":     (130, 0,   130),
    "Happy":    (0,   210, 255),
    "Neutral":  (160, 160, 160),
    "Sad":      (200, 100, 0  ),
    "Surprise": (0,   165, 255),
}
DECEPTION_COLORS = {
    "Low":         (0,   200, 0  ),
    "Medium":      (0,   165, 255),
    "High":        (0,   0,   220),
    "Analyzing...": (160, 160, 160),
}

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


class Dashboard:
    def __init__(self):
        self.prob_history = {e: collections.deque([0.0]*HISTORY_LEN,
                              maxlen=HISTORY_LEN) for e in EMOTIONS}
        self.dec_history  = collections.deque([0.0]*HISTORY_LEN,
                              maxlen=HISTORY_LEN)
        self.dec_label    = "Analyzing..."
        self.top_emotion  = "Neutral"
        self.top_conf     = 0.0
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)

    def update(self, all_probs: dict, dec_score: float, dec_label: str,
               top_emotion: str, top_conf: float):
        for e in EMOTIONS:
            self.prob_history[e].append(all_probs.get(e, 0.0))
        self.dec_history.append(dec_score)
        self.dec_label   = dec_label
        self.top_emotion = top_emotion
        self.top_conf    = top_conf
        self._draw()

    def render(self, frame: np.ndarray, results: list) -> np.ndarray:
        output = frame.copy()
        for r in results:
            x, y, w, h = r["bbox"]
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

            emotion   = r.get("emotion",   {})
            ethnicity = r.get("ethnicity", {})
            deception = r.get("deception", {})

            lines = [
                f"Emotion: {emotion.get('label','?')} ({emotion.get('confidence',0):.0%})",
                f"Ethnicity: {ethnicity.get('label','?')} ({ethnicity.get('confidence',0):.0%})",
                f"Deception: {deception.get('label','?')} ({deception.get('score',0):.2f})",
            ]
            for i, line in enumerate(lines):
                cv2.putText(output, line, (x, y - 10 - i*18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Also update the rolling dashboard if emotion scores available
            scores = emotion.get("scores", {})
            if scores:
                self.update(
                    all_probs=scores,
                    dec_score=deception.get("score", 0.0),
                    dec_label=deception.get("label", "Analyzing..."),
                    top_emotion=emotion.get("label", "Neutral"),
                    top_conf=emotion.get("confidence", 0.0),
                )

        return output

    def _draw(self):
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 25)

        self._draw_title(canvas)
        self._draw_emotion_lines(canvas)
        self._draw_deception_gauge(canvas)
        self._draw_legend(canvas)

        cv2.imshow(WIN_NAME, canvas)

    def _draw_title(self, canvas):
        label = f"Emotion: {self.top_emotion}  ({self.top_conf:.0%})"
        cv2.putText(canvas, label, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

    def _draw_emotion_lines(self, canvas):
        GRAPH_X1, GRAPH_X2 = 20, WIN_W - 20
        GRAPH_Y1, GRAPH_Y2 = 50, 280
        GRAPH_W = GRAPH_X2 - GRAPH_X1
        GRAPH_H = GRAPH_Y2 - GRAPH_Y1

        cv2.rectangle(canvas, (GRAPH_X1, GRAPH_Y1),
                      (GRAPH_X2, GRAPH_Y2), (40, 40, 40), -1)
        cv2.rectangle(canvas, (GRAPH_X1, GRAPH_Y1),
                      (GRAPH_X2, GRAPH_Y2), (80, 80, 80), 1)

        for pct in [0.25, 0.50, 0.75]:
            gy = int(GRAPH_Y2 - pct * GRAPH_H)
            cv2.line(canvas, (GRAPH_X1, gy), (GRAPH_X2, gy), (60, 60, 60), 1)
            cv2.putText(canvas, f"{int(pct*100)}%", (GRAPH_X2+2, gy+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        for emotion in EMOTIONS:
            hist  = list(self.prob_history[emotion])
            color = EMOTION_COLORS_BGR[emotion]
            pts   = []
            for i, v in enumerate(hist):
                x = GRAPH_X1 + int(i * GRAPH_W / HISTORY_LEN)
                y = int(GRAPH_Y2 - v * GRAPH_H)
                pts.append((x, y))
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i-1], pts[i], color, 1, cv2.LINE_AA)

        cv2.putText(canvas, "Emotion probabilities (rolling)",
                    (GRAPH_X1, GRAPH_Y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    def _draw_deception_gauge(self, canvas):
        GRAPH_X1, GRAPH_X2 = 20, WIN_W - 20
        GRAPH_Y1, GRAPH_Y2 = 300, 420
        GRAPH_W = GRAPH_X2 - GRAPH_X1
        GRAPH_H = GRAPH_Y2 - GRAPH_Y1

        cv2.rectangle(canvas, (GRAPH_X1, GRAPH_Y1),
                      (GRAPH_X2, GRAPH_Y2), (40, 40, 40), -1)
        cv2.rectangle(canvas, (GRAPH_X1, GRAPH_Y1),
                      (GRAPH_X2, GRAPH_Y2), (80, 80, 80), 1)

        hist  = list(self.dec_history)
        color = DECEPTION_COLORS.get(self.dec_label, (160, 160, 160))
        pts   = []
        for i, v in enumerate(hist):
            x = GRAPH_X1 + int(i * GRAPH_W / HISTORY_LEN)
            y = int(GRAPH_Y2 - v * GRAPH_H)
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], color, 2, cv2.LINE_AA)

        cur_score = hist[-1] if hist else 0.0
        label_txt = f"Deception: {self.dec_label}  ({cur_score:.2f})"
        cv2.putText(canvas, label_txt, (GRAPH_X1, GRAPH_Y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        for pct, name in [(0.35, "Low/Med"), (0.60, "Med/High")]:
            gy = int(GRAPH_Y2 - pct * GRAPH_H)
            cv2.line(canvas, (GRAPH_X1, gy), (GRAPH_X2, gy), (80, 80, 80), 1)

    def _draw_legend(self, canvas):
        x, y = 20, 445
        for emotion in EMOTIONS:
            color = EMOTION_COLORS_BGR[emotion]
            cv2.rectangle(canvas, (x, y), (x+12, y+10), color, -1)
            cv2.putText(canvas, emotion[:3], (x+15, y+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
            x += 72

    def close(self):
        cv2.destroyWindow(WIN_NAME)