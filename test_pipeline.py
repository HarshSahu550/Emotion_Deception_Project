# ── Test 6: ethnicity module ───────────────────────────
print("[ ethnicity_module ]")
try:
    from ethnicity_module.ethnicity_model import predict_ethnicity

    check("ethnicity_model imports", True)

    # Test with a dummy face — API must be running for this to pass
    dummy_face = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    label, conf = predict_ethnicity(dummy_face)

    check("predict_ethnicity returns tuple",
          isinstance(label, str) and isinstance(conf, float))
    check("label is valid ethnicity or Unknown",
          label in ["White", "Black", "Asian", "Indian", "Others", "Unknown"],
          f"got '{label}'")
    check("confidence in [0,1]", 0.0 <= conf <= 1.0, f"got {conf:.3f}")

except Exception as e:
    check("ethnicity_module", False, str(e))

print()

# ── Test 7: full pipeline with ethnicity ───────────────
print("[ integration with ethnicity ]")
try:
    from face_module.face_detector import detect_faces
    from emotion_module.emotion_inference import predict_emotion, predict_emotion_all
    from deception_module.deception_logic import update_and_score, reset
    from ethnicity_module.ethnicity_model import predict_ethnicity

    reset()
    fake_face_crop = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)

    emotion, conf    = predict_emotion(fake_face_crop)
    all_probs        = predict_emotion_all(fake_face_crop)
    score, dec_label = update_and_score(all_probs)
    ethnicity, e_conf = predict_ethnicity(fake_face_crop)

    check("full pipeline with ethnicity runs", True)
    check("ethnicity output valid",
          ethnicity in ["White", "Black", "Asian", "Indian", "Others", "Unknown"])

except Exception as e:
    check("integration with ethnicity", False, str(e))

print()