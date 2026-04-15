"""
test_pipeline.py
Run from project root: python test_pipeline.py
Make sure the ethnicity Flask API is running before executing:
    python ethnicity_module/app.py
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

print("\n" + "="*55)
print(" EMOTION DECEPTION PROJECT — PIPELINE TESTS")
print("="*55 + "\n")

# ── Test 1: config ─────────────────────────────────────
print("[ config ]")
try:
    from config import (
        EMOTION_LABELS, IMG_SIZE, NUM_CLASSES,
        CONFIDENCE_THRESHOLD, EMOTION_MODEL_H5,
        ETHNICITY_API_URL, ETHNICITY_LABELS
    )
    check("config imports",              True)
    check("7 emotion labels",            len(EMOTION_LABELS) == 7, str(EMOTION_LABELS))
    check("IMG_SIZE is (48,48)",         IMG_SIZE == (48, 48))
    check("emotion model file exists",   os.path.exists(EMOTION_MODEL_H5), EMOTION_MODEL_H5)
    check("ETHNICITY_API_URL present",   isinstance(ETHNICITY_API_URL, str) and len(ETHNICITY_API_URL) > 0)
    check("5 ethnicity labels",          len(ETHNICITY_LABELS) == 5, str(ETHNICITY_LABELS))
except Exception as e:
    check("config imports", False, str(e))

print()

# ── Test 2: face detector ──────────────────────────────
print("[ face_module ]")
try:
    from face_module.face_detector import detect_faces
    check("face_detector imports", True)

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    check("returns list on blank frame", isinstance(detect_faces(blank), list))

    white = np.ones((480, 640, 3), dtype=np.uint8) * 255
    check("returns list on white frame", isinstance(detect_faces(white), list))

except Exception as e:
    check("face_detector", False, str(e))

print()

# ── Test 3: emotion inference ──────────────────────────
print("[ emotion_module ]")
try:
    from emotion_module.emotion_inference import predict_emotion, predict_emotion_all
    check("emotion_inference imports", True)

    dummy_face = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)

    label, conf = predict_emotion(dummy_face)
    check("predict_emotion returns tuple",
          isinstance(label, str) and isinstance(conf, float))
    check("label is valid",
          label in EMOTION_LABELS + ["Uncertain", "No face"], f"got '{label}'")
    check("confidence in [0,1]", 0.0 <= conf <= 1.0, f"got {conf:.3f}")

    all_probs = predict_emotion_all(dummy_face)
    check("predict_emotion_all returns dict",  isinstance(all_probs, dict))
    check("dict has 7 emotions",               len(all_probs) == 7)
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
    dummy_probs = {l: 1/7 for l in EMOTION_LABELS}
    for _ in range(10):
        score, label = update_and_score(dummy_probs)

    check("score is float",   isinstance(score, float))
    check("score in [0,1]",   0.0 <= score <= 1.0, f"got {score:.3f}")
    check("label is valid",   label in ["Low", "Medium", "High", "Analyzing..."])

    reset()
    score2, label2 = update_and_score(dummy_probs)
    check("reset clears history", label2 == "Analyzing...")

except Exception as e:
    check("deception_logic", False, str(e))

print()

# ── Test 5: ethnicity module ───────────────────────────
print("[ ethnicity_module ]")
try:
    from ethnicity_module.ethnicity_model import predict_ethnicity
    check("ethnicity_model imports", True)

    dummy_face = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    eth_label, eth_conf = predict_ethnicity(dummy_face)

    check("predict_ethnicity returns tuple",
          isinstance(eth_label, str) and isinstance(eth_conf, float))
    check("label is valid ethnicity or Unknown",
          eth_label in ["White", "Black", "Asian", "Indian", "Others", "Unknown"],
          f"got '{eth_label}'")
    check("confidence in [0,1]", 0.0 <= eth_conf <= 1.0, f"got {eth_conf:.3f}")

    # API connectivity check
    if eth_label == "Unknown":
        print("       [WARN] Got 'Unknown' — Flask API may not be running.")
        print("              Start it with:  python ethnicity_module/app.py")

except Exception as e:
    check("ethnicity_module", False, str(e))

print()

# ── Test 6: full pipeline integration ─────────────────
print("[ integration ]")
try:
    from face_module.face_detector import detect_faces
    from emotion_module.emotion_inference import predict_emotion, predict_emotion_all
    from deception_module.deception_logic import update_and_score, reset
    from ethnicity_module.ethnicity_model import predict_ethnicity

    reset()
    fake_face = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)

    emotion, conf        = predict_emotion(fake_face)
    all_probs            = predict_emotion_all(fake_face)
    score, dec_label     = update_and_score(all_probs)
    eth_label, eth_conf  = predict_ethnicity(fake_face)

    check("full pipeline runs without error", True)
    check("emotion output valid",
          emotion in EMOTION_LABELS + ["Uncertain", "No face"])
    check("deception output valid",
          dec_label in ["Low", "Medium", "High", "Analyzing..."])
    check("ethnicity output valid",
          eth_label in ["White", "Black", "Asian", "Indian", "Others", "Unknown"])
    check("all three outputs produced simultaneously", True)

except Exception as e:
    check("integration", False, str(e))

print()

# ── Summary ────────────────────────────────────────────
print("="*55)
passed = sum(results)
total  = len(results)
print(f" Results: {passed}/{total} tests passed")
if passed == total:
    print(" All tests passed — pipeline is ready to run")
else:
    print(f" {total - passed} test(s) failed — check output above")
    if passed >= total - 3:
        print(" (ethnicity failures are OK if Flask API is not running)")
print("="*55 + "\n")