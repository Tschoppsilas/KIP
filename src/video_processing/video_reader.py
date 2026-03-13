"""Utilities for loading video frames."""

from pathlib import Path

import cv2
import numpy as np


def read_first_frame(video_path: str | Path) -> np.ndarray:
    """Load and return the first frame of a video as a NumPy array."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    success, frame = capture.read()
    capture.release()

    if not success or frame is None:
        raise ValueError(f"Could not read first frame from: {path}")

    return frame
