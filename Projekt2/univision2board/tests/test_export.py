"""Tests für PNG/PDF-Bildexport und VideoExporter."""

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Taktikboard", "Taktikboard.png"
)
VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Videos", "Muenchenstein_1.mp4"
)


def _board_img():
    img = cv2.imread(BOARD_PATH)
    if img is None:
        pytest.skip("Taktikboard.png nicht gefunden.")
    return img


def _dummy_frame(h=720, w=1280):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# PNG-Export
# ---------------------------------------------------------------------------

class TestExportPNG:
    def test_creates_file(self):
        from src.utils import export_png
        img = _dummy_frame(100, 200)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.png")
            export_png(img, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_content_matches(self):
        """Gelesenes Bild muss dieselbe Grösse wie gespeichertes haben."""
        from src.utils import export_png
        img = _dummy_frame(50, 80)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "board.png")
            export_png(img, path)
            reloaded = cv2.imread(path)
        assert reloaded is not None
        assert reloaded.shape == img.shape

    def test_creates_parent_dir(self):
        from src.utils import export_png
        img = _dummy_frame(50, 50)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "subdir", "out.png")
            export_png(img, path)
            assert os.path.exists(path)

    def test_real_board_export(self):
        from src.utils import export_png
        img = _board_img()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "taktikboard.png")
            export_png(img, path)
            reloaded = cv2.imread(path)
        assert reloaded.shape == img.shape


# ---------------------------------------------------------------------------
# PDF-Export
# ---------------------------------------------------------------------------

class TestExportPDF:
    def test_creates_file(self):
        from src.utils import export_pdf
        img = _dummy_frame(100, 200)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.pdf")
            export_pdf(img, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_pdf_has_pdf_header(self):
        """PDF-Dateien beginnen mit %PDF."""
        from src.utils import export_pdf
        img = _dummy_frame(100, 200)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.pdf")
            export_pdf(img, path, title="Test")
            with open(path, "rb") as f:
                header = f.read(4)
        assert header == b"%PDF"

    def test_real_board_pdf(self):
        from src.utils import export_pdf
        img = _board_img()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "board.pdf")
            export_pdf(img, path)
            assert os.path.getsize(path) > 1000


# ---------------------------------------------------------------------------
# render_board_frame
# ---------------------------------------------------------------------------

class TestRenderBoardFrame:
    def test_output_same_size_as_input(self):
        from src.utils import render_board_frame
        board = _board_img()
        positions = {1: (200.0, 300.0), 2: (600.0, 400.0)}
        teams = {1: 0, 2: 1}
        trajectories = {1: [(150.0, 280.0), (200.0, 300.0)]}
        out = render_board_frame(board, positions, teams, trajectories)
        assert out.shape == board.shape

    def test_does_not_modify_original(self):
        from src.utils import render_board_frame
        board = _board_img()
        original = board.copy()
        render_board_frame(board, {1: (100.0, 100.0)}, {1: 0}, {})
        assert np.array_equal(board, original)

    def test_empty_positions(self):
        from src.utils import render_board_frame
        board = _board_img()
        out = render_board_frame(board, {}, {}, {})
        assert out.shape == board.shape


# ---------------------------------------------------------------------------
# VideoExporter
# ---------------------------------------------------------------------------

class TestVideoExporter:
    def test_creates_video_file(self):
        from src.utils import VideoExporter
        board = _board_img()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.mp4")
            with VideoExporter(path, fps=30.0, split_view=False) as exp:
                for _ in range(5):
                    exp.write_frame(board)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_video_readable_by_opencv(self):
        """Das erzeugte Video muss von OpenCV gelesen werden können."""
        from src.utils import VideoExporter
        board = _board_img()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.mp4")
            with VideoExporter(path, fps=30.0, split_view=False) as exp:
                for _ in range(3):
                    exp.write_frame(board)
            cap = cv2.VideoCapture(path)
            assert cap.isOpened()
            ret, frame = cap.read()
            cap.release()
        assert ret
        assert frame is not None

    def test_split_view_doubles_width(self):
        """Split-View muss Video-Frame + Board nebeneinander haben."""
        from src.utils import VideoExporter
        board = _board_img()
        video_frame = _dummy_frame(board.shape[0], board.shape[1])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "split.mp4")
            with VideoExporter(path, fps=30.0, split_view=True) as exp:
                exp.write_frame(board, video_frame)
            cap = cv2.VideoCapture(path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
        assert w == board.shape[1] * 2

    def test_frame_count_correct(self):
        from src.utils import VideoExporter
        board = _dummy_frame(100, 200)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "count.mp4")
            exp = VideoExporter(path, fps=25.0, split_view=False)
            for _ in range(10):
                exp.write_frame(board)
            exp.close()
            assert exp._frame_count == 10

    def test_export_with_real_video_frames(self):
        """Exportiert 10 echte Video-Frames mit Board-Overlay."""
        if not os.path.exists(VIDEO_PATH):
            pytest.skip("Testvideo nicht gefunden.")
        from src.utils import VideoExporter, render_board_frame
        board = _board_img()
        positions = {1: (300.0, 200.0), 2: (700.0, 500.0)}
        teams = {1: 0, 2: 1}
        trajectories = {1: [(280.0, 190.0), (300.0, 200.0)]}

        cap = cv2.VideoCapture(VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "annotated.mp4")
            with VideoExporter(path, fps=fps, split_view=True) as exp:
                for _ in range(10):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    board_frame = render_board_frame(board, positions, teams, trajectories)
                    exp.write_frame(board_frame, frame)
            cap.release()
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            cap2 = cv2.VideoCapture(path)
            assert cap2.isOpened()
            cap2.release()
