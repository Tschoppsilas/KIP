"""Object detection module (Phase 3) – YOLOv11-based player/ball detection."""

from .detection import DetectionResult, CLASS_PLAYER, CLASS_GOALKEEPER, CLASS_BALL
from .detector import Detector, DEFAULT_MODEL

__all__ = [
    "DetectionResult",
    "CLASS_PLAYER",
    "CLASS_GOALKEEPER",
    "CLASS_BALL",
    "Detector",
    "DEFAULT_MODEL",
]
