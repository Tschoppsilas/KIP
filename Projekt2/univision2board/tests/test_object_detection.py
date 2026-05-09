"""Tests für DetectionResult und YOLODetector."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "finetune", "runs", "train", "weights", "best.pt"
)
VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Videos", "Muenchenstein_1.mp4"
)


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------

class TestDetectionResult:
    def _make_result(self):
        from src.object_detection import DetectionResult, CLASS_PLAYER, CLASS_GOALKEEPER
        dr = DetectionResult(frame_idx=0)
        dr.boxes = [
            np.array([10, 20, 50, 80], dtype=np.float32),  # player
            np.array([200, 100, 260, 200], dtype=np.float32),  # goalkeeper
        ]
        dr.confidences = [0.9, 0.75]
        dr.class_ids = [CLASS_PLAYER, CLASS_GOALKEEPER]
        dr.class_names = ["player", "goalkeeper"]
        return dr

    def test_len(self):
        dr = self._make_result()
        assert len(dr) == 2

    def test_center(self):
        dr = self._make_result()
        cx, cy = dr.center(0)
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(50.0)

    def test_players_only(self):
        from src.object_detection import CLASS_PLAYER
        dr = self._make_result()
        players = dr.players_only()
        assert len(players) == 1
        assert players.class_ids[0] == CLASS_PLAYER

    def test_filter_by_class(self):
        from src.object_detection import CLASS_GOALKEEPER
        dr = self._make_result()
        gk = dr.filter_by_class(CLASS_GOALKEEPER)
        assert len(gk) == 1
        assert gk.class_names[0] == "goalkeeper"

    def test_empty_result(self):
        from src.object_detection import DetectionResult
        dr = DetectionResult(frame_idx=5)
        assert len(dr) == 0
        assert dr.players_only() is not None


# ---------------------------------------------------------------------------
# YOLODetector
# ---------------------------------------------------------------------------

class TestYOLODetector:
    def test_model_loads(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip("best.pt nicht gefunden.")
        from src.object_detection import YOLODetector
        det = YOLODetector(MODEL_PATH)
        assert det is not None

    def test_detect_returns_detection_result(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip("best.pt nicht gefunden.")
        from src.object_detection import YOLODetector, DetectionResult
        import cv2
        if not os.path.exists(VIDEO_PATH):
            pytest.skip("Testvideo nicht gefunden.")
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        cap.release()
        assert ret

        det = YOLODetector(MODEL_PATH)
        result = det.detect(frame, frame_idx=0)
        assert isinstance(result, DetectionResult)
        assert result.frame_idx == 0

    def test_detect_finds_players(self):
        """Das fine-getunede Modell soll auf einem Spielfeld-Frame Spieler finden."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("best.pt nicht gefunden.")
        if not os.path.exists(VIDEO_PATH):
            pytest.skip("Testvideo nicht gefunden.")
        import cv2
        from src.object_detection import YOLODetector

        cap = cv2.VideoCapture(VIDEO_PATH)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        cap.release()
        assert ret

        det = YOLODetector(MODEL_PATH, conf=0.2)
        result = det.detect(frame)
        assert len(result) > 0, "Kein Spieler im Frame erkannt – Modell oder conf prüfen."

    def test_roi_filter(self):
        """Detektionen außerhalb des ROI dürfen nicht zurückgegeben werden."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("best.pt nicht gefunden.")
        import cv2
        from src.object_detection import YOLODetector

        if not os.path.exists(VIDEO_PATH):
            pytest.skip("Testvideo nicht gefunden.")
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        cap.release()
        assert ret

        h, w = frame.shape[:2]
        # ROI: nur linke obere Ecke – sehr wenig Spielfeld
        roi = (0, 0, w // 4, h // 4)
        det_full = YOLODetector(MODEL_PATH, conf=0.2)
        det_roi = YOLODetector(MODEL_PATH, conf=0.2, roi=roi)

        full = det_full.detect(frame)
        restricted = det_roi.detect(frame)
        # ROI-Version darf nicht mehr Detektionen liefern als die uneingeschränkte
        assert len(restricted) <= len(full)

    def test_output_format(self):
        """Boxes müssen [x1, y1, x2, y2], confidences in [0,1] liegen."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("best.pt nicht gefunden.")
        if not os.path.exists(VIDEO_PATH):
            pytest.skip("Testvideo nicht gefunden.")
        import cv2
        from src.object_detection import YOLODetector

        cap = cv2.VideoCapture(VIDEO_PATH)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        cap.release()

        det = YOLODetector(MODEL_PATH, conf=0.2)
        result = det.detect(frame)

        for box, conf, cls_name in zip(result.boxes, result.confidences, result.class_names):
            assert box.shape == (4,), "Box muss 4 Werte haben."
            assert box[0] < box[2], "x1 muss kleiner x2 sein."
            assert box[1] < box[3], "y1 muss kleiner y2 sein."
            assert 0.0 <= conf <= 1.0, "Confidence muss zwischen 0 und 1 liegen."
            assert isinstance(cls_name, str)
