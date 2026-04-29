"""
deception_module/deception_logic.py

Rule-based deception scoring using emotion probabilities.
No separate model needed — works entirely from emotion_module output.

Deception score is 0.0 to 1.0:
  0.0 - 0.35  → Low
  0.35 - 0.60 → Medium  
  0.60+       → High
"""

from collections import deque
import numpy as np

# ── Tunable weights ────────────────────────────────────
SUPPRESSION_WEIGHT  = 0.45   # hiding fear/disgust behind neutral/happy
FLICKERING_WEIGHT   = 0.30   # rapid emotion switching
FLATNESS_WEIGHT     = 0.25   # unnaturally stable for too long

HISTORY_LEN         = 30     # frames to keep in memory (~1 sec at 30fps)
FLICKER_THRESHOLD   = 0.50   # min change in top emotion to count as a switch
FLATNESS_THRESHOLD  = 20     # frames of same emotion = suspiciously flat

# Emotions that are typically suppressed when lying
SUPPRESSED_EMOTIONS = {"Fear", "Disgust", "Angry"}
# Emotions used as a "mask" when lying
MASK_EMOTIONS       = {"Neutral", "Happy"}


class DeceptionAnalyzer:
    """
    Maintains a rolling window of emotion history per face.
    Call update() every frame, get_score() to read current deception level.
    """

    def __init__(self):
        self.history  = deque(maxlen=HISTORY_LEN)  # list of {emotion: prob} dicts
        self.top_history = deque(maxlen=HISTORY_LEN)  # list of top emotion strings

    def update(self, all_probs: dict):
        """Feed in the full emotion probability dict for this frame."""
        self.history.append(all_probs)
        top = max(all_probs, key=all_probs.get)
        self.top_history.append(top)

    def get_score(self) -> tuple[float, str]:
        """
        Returns (score 0-1, label string).
        Needs at least 5 frames of history to be meaningful.
        """
        if len(self.history) < 5:
            return 0.0, "Analyzing..."

        suppression = self._suppression_score()
        flickering  = self._flickering_score()
        flatness    = self._flatness_score()

        score = (
            suppression * SUPPRESSION_WEIGHT +
            flickering  * FLICKERING_WEIGHT  +
            flatness    * FLATNESS_WEIGHT
        )
        score = float(np.clip(score, 0.0, 1.0))

        if score < 0.35:
            label = "Low"
        elif score < 0.60:
            label = "Medium"
        else:
            label = "High"

        return score, label

    def _suppression_score(self) -> float:
        """
        High score when suppressed emotions (fear/disgust/angry) have
        significant probability but the top emotion is a mask (neutral/happy).
        """
        scores = []
        for probs in self.history:
            top = max(probs, key=probs.get)
            if top in MASK_EMOTIONS:
                suppressed_sum = sum(
                    probs.get(e, 0.0) for e in SUPPRESSED_EMOTIONS
                )
                scores.append(suppressed_sum)
            else:
                scores.append(0.0)
        return float(np.mean(scores)) * 2.5   # scale to ~0-1 range

    def _flickering_score(self) -> float:
        """
        High score when top emotion switches rapidly across frames.
        """
        if len(self.top_history) < 2:
            return 0.0
        switches = sum(
            1 for a, b in zip(self.top_history, list(self.top_history)[1:])
            if a != b
        )
        return switches / max(len(self.top_history) - 1, 1)

    def _flatness_score(self) -> float:
        """
        High score when the same emotion persists unnaturally long
        with very high confidence (robotic stillness).
        """
        if len(self.top_history) < FLATNESS_THRESHOLD:
            return 0.0

        recent = list(self.top_history)[-FLATNESS_THRESHOLD:]
        if len(set(recent)) == 1:
            # all same emotion — check avg confidence
            emotion = recent[0]
            avg_conf = np.mean([p.get(emotion, 0) for p in
                                list(self.history)[-FLATNESS_THRESHOLD:]])
            return float(avg_conf)
        return 0.0

    def reset(self):
        self.history.clear()
        self.top_history.clear()


# ── Simple stateless API (for single-frame use) ────────
_analyzer = DeceptionAnalyzer()

def update_and_score(all_probs: dict) -> tuple[float, str]:
    """
    Feed one frame's emotion probs, get back (score, label).
    Uses a global analyzer — fine for single-face use.
    """
    _analyzer.update(all_probs)
    return _analyzer.get_score()

def reset():
    _analyzer.reset()
    
class DeceptionScorer:
    def score(self, emotion: dict) -> dict:
        label = emotion.get("label", "Neutral")
        confidence = emotion.get("confidence", 0.0)
        # Map single emotion result to a prob dict for the analyzer
        probs = {e: 0.0 for e in SUPPRESSED_EMOTIONS | MASK_EMOTIONS |
                 {"Sad", "Surprise", "Happy", "Neutral", "Angry", "Fear", "Disgust"}}
        probs[label] = confidence
        score, dec_label = update_and_score(probs)
        return {"label": dec_label, "score": score}