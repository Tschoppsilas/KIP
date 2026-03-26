"""Detection result data structures for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


# Einheitliche Klassennamen fuer unsere Domain
CLASS_PLAYER = "player"
CLASS_GOALKEEPER = "goalkeeper"
CLASS_BALL = "ball"

# COCO-Klassen-IDs die wir verwenden
COCO_PERSON = 0
COCO_SPORTS_BALL = 32


@dataclass
class DetectionResult:
    """Single object detection from one video frame.

    Attributes:
        bbox:        Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        class_id:    Internal domain class ID (0=player, 1=goalkeeper, 2=ball).
        class_name:  Human-readable class label.
        confidence:  Detection confidence in [0, 1].
        frame_index: Index of the source frame (optional).
    """

    CLASS_PLAYER: ClassVar[int] = 0
    CLASS_GOALKEEPER: ClassVar[int] = 1
    CLASS_BALL: ClassVar[int] = 2

    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    frame_index: int = -1

    @property
    def center(self) -> tuple[float, float]:
        """Return the center (cx, cy) of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_dict(self) -> dict:
        """Serialize to a plain dict (for JSON export or logging)."""
        return {
            "bbox": list(self.bbox),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "center": list(self.center),
            "frame_index": self.frame_index,
        }
