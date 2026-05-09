"""Tests für VideoLoader und HomographyTransformer."""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Videos", "Muenchenstein_1.mp4")


# ---------------------------------------------------------------------------
# VideoLoader
# ---------------------------------------------------------------------------

class TestVideoLoader:
    def test_open_and_close(self):
        from src.video_processing import VideoLoader
        if not os.path.exists(VIDEO_PATH):
            pytest.skip("Testvideo nicht vorhanden.")
        loader = VideoLoader(VIDEO_PATH).open()
        assert loader.fps > 0
        assert loader.frame_count > 0
        assert loader.width > 0
        assert loader.height > 0
        loader.close()

    def test_context_manager(self):
        from src.video_processing import VideoLoader
        if not os.path.exists(VIDEO_PATH):
            pytest.skip("Testvideo nicht vorhanden.")
        with VideoLoader(VIDEO_PATH) as loader:
            frame = loader.read_frame(0)
        assert frame is not None
        assert frame.shape[0] > 0

    def test_read_specific_frame(self):
        from src.video_processing import VideoLoader
        if not os.path.exists(VIDEO_PATH):
            pytest.skip("Testvideo nicht vorhanden.")
        with VideoLoader(VIDEO_PATH) as loader:
            frame_0 = loader.read_frame(0)
            frame_5 = loader.read_frame(5)
        assert frame_0 is not None
        assert frame_5 is not None

    def test_frames_iterator(self):
        from src.video_processing import VideoLoader
        if not os.path.exists(VIDEO_PATH):
            pytest.skip("Testvideo nicht vorhanden.")
        with VideoLoader(VIDEO_PATH) as loader:
            frames = list(loader.frames(step=30))
        assert len(frames) > 0
        idx, frame = frames[0]
        assert idx == 0
        assert frame is not None

    def test_invalid_path_raises(self):
        from src.video_processing import VideoLoader
        with pytest.raises(IOError):
            VideoLoader("/nicht/vorhanden.mp4").open()


# ---------------------------------------------------------------------------
# HomographyTransformer
# ---------------------------------------------------------------------------

# Einfaches Testfeld: Quadrat im Video → Rechteck auf Board
SRC = [(100, 100), (500, 100), (500, 400), (100, 400)]
DST = [(200, 150), (800, 150), (800, 600), (200, 600)]


class TestHomographyTransformer:
    def test_compute_requires_4_points(self):
        from src.video_processing import HomographyTransformer
        with pytest.raises(ValueError):
            HomographyTransformer().compute(SRC[:3], DST[:3])

    def test_compute_succeeds(self):
        from src.video_processing import HomographyTransformer
        t = HomographyTransformer().compute(SRC, DST)
        assert t.matrix is not None
        assert t.matrix.shape == (3, 3)

    def test_transform_point_within_src_region(self):
        """Ein Punkt exakt aus der Quelle muss nahe am Zielpunkt landen."""
        from src.video_processing import HomographyTransformer
        t = HomographyTransformer().compute(SRC, DST)
        tx, ty = t.transform_point(*SRC[0])
        assert abs(tx - DST[0][0]) < 2.0
        assert abs(ty - DST[0][1]) < 2.0

    def test_save_and_load(self):
        from src.video_processing import HomographyTransformer
        t = HomographyTransformer().compute(SRC, DST)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            t.save(tmp_path)
            loaded = HomographyTransformer.load(tmp_path)
            tx, ty = loaded.transform_point(*SRC[2])
            assert abs(tx - DST[2][0]) < 2.0
            assert abs(ty - DST[2][1]) < 2.0
        finally:
            os.unlink(tmp_path)

    def test_json_contains_points(self):
        from src.video_processing import HomographyTransformer
        t = HomographyTransformer().compute(SRC, DST)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            t.save(tmp_path)
            with open(tmp_path) as f:
                data = json.load(f)
            assert "src_points" in data
            assert "dst_points" in data
            assert len(data["src_points"]) == 4
        finally:
            os.unlink(tmp_path)
