"""Tests for Phase 2 calibration save/load and template handling."""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_processing.calibration import (
    load_calibration,
    load_homography_from_file,
    load_template,
    list_templates,
    save_calibration,
    save_template,
)

_SRC = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
_DST = [(0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)]


class TestSaveLoadCalibration(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "calib.json"
            save_calibration(_SRC, _DST, p)
            src2, dst2 = load_calibration(p)
            self.assertEqual(src2, _SRC)
            self.assertEqual(dst2, _DST)

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nested" / "deep" / "calib.json"
            save_calibration(_SRC, _DST, p)
            self.assertTrue(p.exists())

    def test_metadata_persisted(self) -> None:
        import json
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "calib.json"
            save_calibration(_SRC, _DST, p, metadata={"video": "test.mp4"})
            data = json.loads(p.read_text())
            self.assertEqual(data["metadata"]["video"], "test.mp4")

    def test_load_homography_returns_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "calib.json"
            save_calibration(_SRC, _DST, p)
            H = load_homography_from_file(p)
            self.assertEqual(H.shape, (3, 3))


class TestTemplates(unittest.TestCase):
    def test_save_and_load_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save_template("kamera_a", _SRC, _DST, templates_dir=tmp)
            src2, dst2 = load_template("kamera_a", templates_dir=tmp)
            self.assertEqual(src2, _SRC)
            self.assertEqual(dst2, _DST)

    def test_list_templates_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_templates(tmp), [])

    def test_list_templates_finds_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save_template("kamera_a", _SRC, _DST, templates_dir=tmp)
            save_template("kamera_b", _SRC, _DST, templates_dir=tmp)
            names = list_templates(tmp)
            self.assertIn("kamera_a", names)
            self.assertIn("kamera_b", names)
