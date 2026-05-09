"""Utilities for loading and iterating video frames."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, NamedTuple

import cv2
import numpy as np


class VideoInfo(NamedTuple):
    width: int
    height: int
    fps: float
    frame_count: int


def _open_capture(video_path: Path) -> cv2.VideoCapture:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    return capture


def get_video_info(video_path: str | Path) -> VideoInfo:
    """Return metadata (width, height, fps, frame_count) for a video file."""
    capture = _open_capture(Path(video_path))
    info = VideoInfo(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=capture.get(cv2.CAP_PROP_FPS),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    return info


def read_first_frame(video_path: str | Path) -> np.ndarray:
    """Load and return the first frame of a video as a NumPy array."""
    capture = _open_capture(Path(video_path))
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        raise ValueError(f"Could not read first frame from: {video_path}")
    return frame


def iter_frames(
    video_path: str | Path,
    max_frames: int | None = None,
    step: int = 1,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield (frame_index, frame_bgr) tuples from a video file.

    Args:
        video_path: Path to the video file.
        max_frames: Stop after this many yielded frames (None = entire video).
        step: Yield only every nth frame (1 = every frame).
    """
    capture = _open_capture(Path(video_path))
    frame_index = 0
    yielded = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            if frame_index % step == 0:
                yield frame_index, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()
