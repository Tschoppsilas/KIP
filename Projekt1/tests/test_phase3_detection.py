"""Tests for Phase 3: object detection data structures and detector."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from object_detection.detection import (
    CLASS_BALL,
    CLASS_GOALKEEPER,
    CLASS_PLAYER,
    COCO_PERSON,
    COCO_SPORTS_BALL,
    DetectionResult,
)


# ─── DetectionResult Tests ────────────────────────────────────────────────────

class TestDetectionResult(unittest.TestCase):

    def _make(self, x1=10, y1=20, x2=110, y2=220, cls_id=0,
              name="player", conf=0.85, fidx=5):
        return DetectionResult(
            bbox=(x1, y1, x2, y2),
            class_id=cls_id,
            class_name=name,
            confidence=conf,
            frame_index=fidx,
        )

    def test_center_computed_correctly(self):
        d = self._make(x1=0, y1=0, x2=100, y2=200)
        self.assertAlmostEqual(d.center[0], 50.0)
        self.assertAlmostEqual(d.center[1], 100.0)

    def test_width_and_height(self):
        d = self._make(x1=10, y1=20, x2=110, y2=220)
        self.assertAlmostEqual(d.width, 100.0)
        self.assertAlmostEqual(d.height, 200.0)

    def test_area(self):
        d = self._make(x1=0, y1=0, x2=50, y2=80)
        self.assertAlmostEqual(d.area, 4000.0)

    def test_as_dict_keys(self):
        d = self._make()
        result = d.as_dict()
        for key in ("bbox", "class_id", "class_name", "confidence", "center", "frame_index"):
            self.assertIn(key, result)

    def test_as_dict_values(self):
        d = self._make(conf=0.9123)
        result = d.as_dict()
        self.assertEqual(result["class_name"], "player")
        self.assertEqual(result["confidence"], 0.9123)

    def test_class_constants(self):
        self.assertEqual(DetectionResult.CLASS_PLAYER,     0)
        self.assertEqual(DetectionResult.CLASS_GOALKEEPER, 1)
        self.assertEqual(DetectionResult.CLASS_BALL,       2)

    def test_player_detection(self):
        d = DetectionResult(
            bbox=(100.0, 50.0, 160.0, 200.0),
            class_id=DetectionResult.CLASS_PLAYER,
            class_name="player",
            confidence=0.75,
        )
        self.assertEqual(d.class_name, "player")
        self.assertGreater(d.confidence, 0)

    def test_goalkeeper_detection(self):
        d = DetectionResult(
            bbox=(10.0, 10.0, 50.0, 100.0),
            class_id=DetectionResult.CLASS_GOALKEEPER,
            class_name="goalkeeper",
            confidence=0.60,
        )
        self.assertEqual(d.class_name, "goalkeeper")

    def test_ball_detection(self):
        d = DetectionResult(
            bbox=(300.0, 200.0, 320.0, 220.0),
            class_id=DetectionResult.CLASS_BALL,
            class_name="ball",
            confidence=0.30,
        )
        self.assertEqual(d.class_name, "ball")
        self.assertLess(d.width, 30)


# ─── Detector Tests (mit gemocktem YOLO-Modell) ───────────────────────────────

def _make_mock_box(x1, y1, x2, y2, cls_id, conf):
    """Helper: create a mock ultralytics box object."""
    box = MagicMock()
    box.xyxy = [np.array([x1, y1, x2, y2], dtype=np.float32)]
    box.cls   = [np.array(cls_id, dtype=np.float32)]
    box.conf  = [np.array(conf,   dtype=np.float32)]
    return box


def _make_mock_result(boxes):
    result = MagicMock()
    result.boxes = boxes
    return result


class TestDetector(unittest.TestCase):

    def _make_detector(self, detect_ball=False):
        """Build a Detector with a fully mocked YOLO model."""
        with patch("object_detection.detector.YOLO") as MockYOLO:
            mock_model = MagicMock()
            MockYOLO.return_value = mock_model

            from object_detection.detector import Detector
            det = Detector(detect_ball=detect_ball)
            det._model = mock_model
            return det, mock_model

    def test_detect_returns_list(self):
        det, mock_model = self._make_detector()
        mock_model.return_value = [_make_mock_result([])]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        self.assertIsInstance(result, list)

    def test_person_mapped_to_player(self):
        det, mock_model = self._make_detector()
        mock_model.return_value = [_make_mock_result(
            [_make_mock_box(10, 20, 80, 200, COCO_PERSON, 0.85)]
        )]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].class_name, "player")
        self.assertAlmostEqual(results[0].confidence, 0.85, places=2)

    def test_ball_excluded_when_disabled(self):
        det, mock_model = self._make_detector(detect_ball=False)
        mock_model.return_value = [_make_mock_result(
            [_make_mock_box(100, 100, 120, 120, COCO_SPORTS_BALL, 0.70)]
        )]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        self.assertEqual(len(results), 0)

    def test_ball_included_when_enabled(self):
        det, mock_model = self._make_detector(detect_ball=True)
        mock_model.return_value = [_make_mock_result(
            [_make_mock_box(100, 100, 120, 120, COCO_SPORTS_BALL, 0.30)]
        )]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].class_name, "ball")

    def test_low_confidence_filtered(self):
        det, mock_model = self._make_detector()
        # Confidence below default threshold (0.40)
        mock_model.return_value = [_make_mock_result(
            [_make_mock_box(10, 20, 80, 200, COCO_PERSON, 0.10)]
        )]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        self.assertEqual(len(results), 0)

    def test_multiple_players_detected(self):
        det, mock_model = self._make_detector()
        mock_model.return_value = [_make_mock_result([
            _make_mock_box(10,  20,  80,  200, COCO_PERSON, 0.90),
            _make_mock_box(200, 50,  260, 250, COCO_PERSON, 0.75),
            _make_mock_box(400, 100, 460, 300, COCO_PERSON, 0.65),
        ])]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.class_name == "player" for r in results))

    def test_frame_index_stored(self):
        det, mock_model = self._make_detector()
        mock_model.return_value = [_make_mock_result(
            [_make_mock_box(10, 20, 80, 200, COCO_PERSON, 0.85)]
        )]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame, frame_index=42)
        self.assertEqual(results[0].frame_index, 42)

    def test_set_conf_threshold(self):
        det, _ = self._make_detector()
        det.set_conf_threshold("player", 0.70)
        self.assertAlmostEqual(det.conf_thresholds["player"], 0.70)

    def test_set_conf_threshold_invalid_class(self):
        det, _ = self._make_detector()
        with self.assertRaises(ValueError):
            det.set_conf_threshold("unknown_class", 0.5)

    def test_detect_sequence_returns_per_frame_lists(self):
        det, mock_model = self._make_detector()
        mock_model.return_value = [_make_mock_result(
            [_make_mock_box(10, 20, 80, 200, COCO_PERSON, 0.85)]
        )]
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        results = det.detect_sequence(frames, start_index=10)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0].frame_index, 10)
        self.assertEqual(results[2][0].frame_index, 12)

    def test_output_format_usable_as_tracking_input(self):
        """Should: Detection-Output ist als Input fuer Tracking nutzbar."""
        det, mock_model = self._make_detector()
        mock_model.return_value = [_make_mock_result([
            _make_mock_box(10, 20, 80, 200, COCO_PERSON, 0.90),
        ])]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        d = results[0]
        # Tracking braucht: bbox (x1,y1,x2,y2), confidence, class_name, center
        self.assertEqual(len(d.bbox), 4)
        self.assertIsInstance(d.confidence, float)
        self.assertIsInstance(d.class_name, str)
        cx, cy = d.center
        self.assertGreater(cx, 0)
        self.assertGreater(cy, 0)


if __name__ == "__main__":
    unittest.main()
