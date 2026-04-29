import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFLITE_PATH = os.path.join(BASE_DIR, "detector.tflite")


class FaceDetector:

    def __init__(self, model_path=None, min_confidence: float = 0.6, model_selection: int = 0):
        base_options = python.BaseOptions(model_asset_path=TFLITE_PATH)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        print("[FaceDetector] MediaPipe face detector ready.")

    def detect(self, bgr_frame) -> list:
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self._detector.detect(mp_image)

        if not results.detections:
            return []

        faces = []
        for detection in results.detections:
            bb = detection.bounding_box
            x = max(0, bb.origin_x)
            y = max(0, bb.origin_y)
            bw = min(bb.width,  w - x)
            bh = min(bb.height, h - y)
            if bw > 0 and bh > 0:
                faces.append((x, y, bw, bh))

        return faces

    def close(self):
        pass