"""Video laden und Frames extrahieren via OpenCV."""

import cv2
import numpy as np
from typing import Iterator, Optional, Tuple

from src.utils import get_logger

logger = get_logger(__name__)


class VideoLoader:
    """Öffnet ein Video und liefert Frames."""

    def __init__(self, path: str):
        self.path = path
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> "VideoLoader":
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"Video konnte nicht geöffnet werden: {self.path}")
        logger.info(
            f"Video geöffnet: {self.path} "
            f"({self.frame_count} Frames, {self.fps:.1f} fps, "
            f"{self.width}x{self.height})"
        )
        return self

    def __enter__(self) -> "VideoLoader":
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap else 0.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    def read_frame(self, index: int) -> Optional[np.ndarray]:
        """Liest einen einzelnen Frame per 0-basiertem Index."""
        if self._cap is None:
            raise RuntimeError("VideoLoader nicht geöffnet.")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        return frame if ret else None

    def frames(self, step: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
        """Iteriert über alle Frames; step überspringt Frames."""
        if self._cap is None:
            raise RuntimeError("VideoLoader nicht geöffnet.")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1
