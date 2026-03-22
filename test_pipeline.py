"""
test_pipeline.py
Run from project root: python test_pipeline.py
Tests each module in isolation, then the full pipeline together.
"""

import sys
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"{status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append(condition)

print("\n" + "="*50)
print(" EMOTION DECEPTION PROJECT — PIPELINE TESTS")
print("="*50 + "\n")

# ── Test 1: config imports ─────────────────────────────
print("[ config ]")
try:
    from config import (
        EMOTION_LABELS, IMG_SIZE, NUM_CLASSES,
        CONFIDENCE_THRESHOLD, EMOTION_MODEL_H5
    )
    check("config imports",        True)
    check("7 emotion labels",      len(EMOTION_LABELS) == 7,
          str(EMOTION_LABELS))
    check("IMG_SIZE is (48,48)",   IMG_SIZE == (48, 48))
    check("model file exists",     os.path.exists(EMOTION_MODEL_H5),
          EMOTION_MODEL_H5)
except Exception as e:
    check("config imports", False, str(e))

print()

# ── Test 2: face detector ──────────────────────────────
print("[ face_module ]")
try:
    from face_module.face_detector import detect_faces
    check("face_detector imports", True)

    # blank frame — no faces expected
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detect_faces(blank)
    check("returns list on blank frame", isinstance(faces, list),
          f"got {type(faces)}")

    # white frame — no faces expected
    white = np.ones((480, 640, 3), dtype=np.uint8) * 255
    faces2 = detect_faces(white)
    check("returns list on white frame", isinstance(faces2, list))

except Exception as e:
    check("face_detector", False, str(e))

print()

# ── Test 3: emotion inference ──────────────────────────
print("[ emotion_module ]")
try:
    from emotion_module.emotion_inference import predict_emotion, predict_emotion_all
    check("emotion_inference imports", True)

    # dummy face — random noise image
    dummy_face = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)

    label, conf = predict_emotion(dummy_face)
    check("predict_emotion returns tuple",  isinstance(label, str) and isinstance(conf, float))
    check("label is valid emotion or uncertain",
          label in EMOTION_LABELS + ["Uncertain", "No face"],
          f"got '{label}'")
    check("confidence in [0,1]",   0.0 <= conf <= 1.0, f"got {conf:.3f}")

    all_probs = predict_emotion_all(dummy_face)
    check("predict_emotion_all returns dict",  isinstance(all_probs, dict))
    check("dict has 7 emotions",   len(all_probs) == 7, f"got {len(all_probs)}")
    check("probabilities sum ~1.0",
          abs(sum(all_probs.values()) - 1.0) < 0.01,
          f"sum={sum(all_probs.values()):.4f}")

except Exception as e:
    check("emotion_inference", False, str(e))

print()

# ── Test 4: deception logic ────────────────────────────
print("[ deception_module ]")
try:
    from deception_module.deception_logic import update_and_score, reset
    check("deception_logic imports", True)

    reset()

    # feed dummy probs for 10 frames
    dummy_probs = {l: 1/7 for l in EMOTION_LABELS}
    for _ in range(10):
        score, label = update_and_score(dummy_probs)

    check("score is float",        isinstance(score, float), f"got {type(score)}")
    check("score in [0,1]",        0.0 <= score <= 1.0,      f"got {score:.3f}")
    check("label is valid",        label in ["Low", "Medium", "High", "Analyzing..."],
          f"got '{label}'")

    reset()
    score2, label2 = update_and_score(dummy_probs)
    check("reset works",           label2 == "Analyzing...")

except Exception as e:
    check("deception_logic", False, str(e))

print()

# ── Test 5: full pipeline (no webcam) ─────────────────
print("[ integration ]")
try:
    import cv2
    from face_module.face_detector import detect_faces
    from emotion_module.emotion_inference import predict_emotion, predict_emotion_all
    from deception_module.deception_logic import update_and_score, reset

    reset()

    # simulate a frame with a fake face crop
    fake_frame     = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    fake_face_crop = np.random.randint(0, 255, (48, 48, 3),   dtype=np.uint8)

    emotion, conf  = predict_emotion(fake_face_crop)
    all_probs      = predict_emotion_all(fake_face_crop)
    score, label   = update_and_score(all_probs)

    check("full pipeline runs without error", True)
    check("emotion output valid",  emotion in EMOTION_LABELS + ["Uncertain"])
    check("deception output valid", label in ["Low","Medium","High","Analyzing..."])

except Exception as e:
    check("integration", False, str(e))

print()

# ── Summary ────────────────────────────────────────────
print("="*50)
passed = sum(results)
total  = len(results)
print(f" Results: {passed}/{total} tests passed")
if passed == total:
    print(" All tests passed — pipeline is ready")
else:
    print(f" {total - passed} test(s) failed — check output above")
print("="*50 + "\n")