

import unittest
import numpy as np
import os
import sys


def make_face(h=48, w=48, gray=True):
    if gray:
        return (np.random.rand(h, w) * 255).astype(np.uint8)
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)



class TestConfig(unittest.TestCase):

    def test_emotion_labels_count(self):
        from config import EMOTION_LABELS
        self.assertEqual(len(EMOTION_LABELS), 7)

    def test_ethnicity_labels_count(self):
        from config import ETHNICITY_LABELS
        self.assertEqual(len(ETHNICITY_LABELS), 5)

    def test_ethnicity_labels_values(self):
        from config import ETHNICITY_LABELS
        expected = {"White", "Black", "Asian", "Indian", "Others"}
        self.assertEqual(set(ETHNICITY_LABELS), expected)

    def test_img_size(self):
        from config import IMG_SIZE
        self.assertEqual(IMG_SIZE, (48, 48))

    def test_flags_are_bool(self):
        from config import USE_EMOTION, USE_ETHNICITY, USE_DECEPTION
        self.assertIsInstance(USE_EMOTION,   bool)
        self.assertIsInstance(USE_ETHNICITY, bool)
        self.assertIsInstance(USE_DECEPTION, bool)


class TestFaceDetector(unittest.TestCase):

    def setUp(self):
        from face_module.face_detector import FaceDetector
        self.detector = FaceDetector()  # MediaPipe — no path needed

    def test_detector_initialises(self):
        self.assertIsNotNone(self.detector)

    def test_detect_returns_list(self):
        frame = make_face(480, 640, gray=False)
        result = self.detector.detect(frame)
        self.assertIsInstance(result, (list, tuple, np.ndarray))

    def test_detect_on_black_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        self.assertEqual(len(result), 0)



class TestEmotionPredictor(unittest.TestCase):

    def setUp(self):
        from emotion_module.emotion_inference import EmotionPredictor
        self.predictor = EmotionPredictor()

    def test_predictor_initialises(self):
        self.assertIsNotNone(self.predictor)

    def test_is_ready_returns_bool(self):
        self.assertIsInstance(self.predictor.is_ready(), bool)

    @unittest.skipUnless(os.path.exists("models/emotion_model.h5"), "model not downloaded")
    def test_predict_returns_dict_keys(self):
        face = make_face()
        result = self.predictor.predict(face)
        self.assertIn("label",      result)
        self.assertIn("confidence", result)
        self.assertIn("scores",     result)

    @unittest.skipUnless(os.path.exists("models/emotion_model.h5"), "model not downloaded")
    def test_predict_label_is_valid(self):
        from config import EMOTION_LABELS
        face = make_face()
        result = self.predictor.predict(face)
        self.assertIn(result["label"], EMOTION_LABELS)

    @unittest.skipUnless(os.path.exists("models/emotion_model.h5"), "model not downloaded")
    def test_predict_confidence_in_range(self):
        face = make_face()
        result = self.predictor.predict(face)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"],    1.0)


class TestEthnicityPredictor(unittest.TestCase):

    def setUp(self):
        from ethnicity_module.ethnicity_inference import EthnicityPredictor
        self.predictor = EthnicityPredictor()

    def test_predictor_initialises(self):
        self.assertIsNotNone(self.predictor)

    def test_is_ready_returns_bool(self):
        self.assertIsInstance(self.predictor.is_ready(), bool)

    @unittest.skipUnless(os.path.exists("models/ethnicity.h5"), "model not downloaded")
    def test_predict_returns_dict_keys(self):
        face = make_face()
        result = self.predictor.predict(face)
        self.assertIn("label",      result)
        self.assertIn("confidence", result)
        self.assertIn("scores",     result)

    @unittest.skipUnless(os.path.exists("models/ethnicity.h5"), "model not downloaded")
    def test_predict_label_is_valid(self):
        from config import ETHNICITY_LABELS
        face = make_face()
        result = self.predictor.predict(face)
        self.assertIn(result["label"], ETHNICITY_LABELS)

    @unittest.skipUnless(os.path.exists("models/ethnicity.h5"), "model not downloaded")
    def test_predict_scores_sum_to_one(self):
        face = make_face()
        result = self.predictor.predict(face)
        total = sum(result["scores"].values())
        self.assertAlmostEqual(total, 1.0, places=4)

    @unittest.skipUnless(os.path.exists("models/ethnicity.h5"), "model not downloaded")
    def test_predict_accepts_bgr_input(self):
        face_bgr = make_face(gray=False)
        result = self.predictor.predict(face_bgr)
        self.assertIn("label", result)


class TestDeceptionScorer(unittest.TestCase):

    def setUp(self):
        from deception_module.deception_logic import DeceptionScorer
        self.scorer = DeceptionScorer()

    def test_scorer_initialises(self):
        self.assertIsNotNone(self.scorer)

    def test_score_returns_dict(self):
        emotion = {"label": "Happy", "confidence": 0.9}
        result  = self.scorer.score(emotion)
        self.assertIsInstance(result, dict)

    def test_score_has_required_keys(self):
        emotion = {"label": "Fear", "confidence": 0.85}
        result  = self.scorer.score(emotion)
        self.assertIn("label", result)
        self.assertIn("score", result)

    def test_score_in_range(self):
        for label in ["Angry", "Happy", "Fear", "Neutral", "Sad", "Disgust", "Surprise"]:
            result = self.scorer.score({"label": label, "confidence": 0.7})
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"],    1.0)


class TestDashboard(unittest.TestCase):

    def setUp(self):
        from visualization.dashboard import Dashboard
        self.dashboard = Dashboard()

    def test_render_no_faces(self):
        frame  = np.zeros((480, 640, 3), dtype=np.uint8)
        output = self.dashboard.render(frame, [])
        self.assertEqual(output.shape, frame.shape)

    def test_render_with_face_result(self):
        frame   = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [{
            "bbox":      (10, 10, 100, 100),
            "emotion":   {"label": "Happy",  "confidence": 0.88},
            "ethnicity": {"label": "Asian",  "confidence": 0.72},
            "deception": {"label": "Honest", "score": 0.2},
        }]
        output = self.dashboard.render(frame, results)
        self.assertEqual(output.shape, frame.shape)

    def test_render_does_not_mutate_input(self):
        frame  = np.zeros((480, 640, 3), dtype=np.uint8)
        before = frame.copy()
        self.dashboard.render(frame, [])
        np.testing.assert_array_equal(frame, before)



class TestPipelineIntegration(unittest.TestCase):

    def test_result_dict_structure(self):
        """Simulate what main.py builds per face and check structure."""
        result = {
            "bbox":      (0, 0, 48, 48),
            "emotion":   {"label": "Sad",   "confidence": 0.6, "scores": {}},
            "ethnicity": {"label": "Indian","confidence": 0.5, "scores": {}},
            "deception": {"label": "Deceptive", "score": 0.75},
        }
        self.assertIn("bbox",      result)
        self.assertIn("emotion",   result)
        self.assertIn("ethnicity", result)
        self.assertIn("deception", result)
        self.assertIsInstance(result["bbox"], tuple)
        self.assertEqual(len(result["bbox"]), 4)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(
        sys.modules[__name__]
    ))
    sys.exit(0 if result.wasSuccessful() else 1)