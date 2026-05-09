"""Basis-Tests: Umgebung und Video-Frame-Einlesen."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_opencv_importable():
    """OpenCV muss importierbar sein."""
    import cv2  # noqa: F401


def test_numpy_importable():
    """NumPy muss importierbar sein."""
    import numpy as np  # noqa: F401

    assert np.__version__


def test_logger_works():
    """Logger muss ohne Fehler instanziierbar sein."""
    from src.utils import get_logger

    log = get_logger("test")
    log.info("Logger funktioniert.")
    assert log is not None


def test_video_frame_readable():
    """Erster Frame eines Testvideos muss lesbar sein."""
    import cv2

    video_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "Videos", "Muenchenstein_1.mp4"
    )
    if not os.path.exists(video_path):
        pytest.skip("Testvideo nicht gefunden – wird beim echten Run benötigt.")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Video konnte nicht geöffnet werden."
    ret, frame = cap.read()
    cap.release()

    assert ret, "Erster Frame konnte nicht gelesen werden."
    assert frame is not None
    assert frame.shape[0] > 0 and frame.shape[1] > 0
