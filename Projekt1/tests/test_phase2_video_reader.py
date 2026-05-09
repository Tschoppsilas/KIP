"""Tests for Phase 2 video loading and frame iteration."""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_processing.video_reader import (
    get_video_info,
    iter_frames,
    read_first_frame,
)


def _make_test_video(path: Path, n_frames: int = 5, width: int = 64, height: int = 48) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (width, height),
    )
    assert writer.isOpened()
    rng = np.random.default_rng(42)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    writer.release()


class TestGetVideoInfo(unittest.TestCase):
    def test_returns_correct_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.avi"
            _make_test_video(p, n_frames=5, width=64, height=48)
            info = get_video_info(p)
            self.assertEqual(info.width, 64)
            self.assertEqual(info.height, 48)
            self.assertGreater(info.fps, 0)
            self.assertGreater(info.frame_count, 0)

    def test_raises_for_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            get_video_info("/nonexistent/video.mp4")


class TestIterFrames(unittest.TestCase):
    def test_yields_all_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.avi"
            _make_test_video(p, n_frames=5)
            frames = list(iter_frames(p))
            self.assertEqual(len(frames), 5)

    def test_max_frames_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.avi"
            _make_test_video(p, n_frames=10)
            frames = list(iter_frames(p, max_frames=3))
            self.assertEqual(len(frames), 3)

    def test_step_skips_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.avi"
            _make_test_video(p, n_frames=10)
            frames = list(iter_frames(p, step=2))
            self.assertEqual(len(frames), 5)
            indices = [idx for idx, _ in frames]
            self.assertEqual(indices, [0, 2, 4, 6, 8])

    def test_frame_shape_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.avi"
            _make_test_video(p, n_frames=3, width=64, height=48)
            for _, frame in iter_frames(p):
                self.assertEqual(frame.shape, (48, 64, 3))
                self.assertEqual(frame.dtype, np.uint8)

    def test_multiple_scenes_independent(self) -> None:
        """Should: Kalibrierung auf mehreren Szenen gegentesten."""
        with tempfile.TemporaryDirectory() as tmp:
            for name, n in [("scene_a.avi", 4), ("scene_b.avi", 7)]:
                p = Path(tmp) / name
                _make_test_video(p, n_frames=n)
                loaded = list(iter_frames(p))
                self.assertEqual(len(loaded), n, f"Falsche Frameanzahl fuer {name}")
