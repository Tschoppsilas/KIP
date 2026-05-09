"""Einheitliches Ausgabeformat für einen Detection-Durchlauf pro Frame."""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# Klassen-IDs aus data.yaml des Fine-Tune-Trainings
CLASS_PLAYER = 0
CLASS_GOALKEEPER = 1
CLASS_BALL = 2
CLASS_REFEREE = 3

CLASS_NAMES = {
    CLASS_PLAYER: "player",
    CLASS_GOALKEEPER: "goalkeeper",
    CLASS_BALL: "ball",
    CLASS_REFEREE: "referee",
}


@dataclass
class DetectionResult:
    """
    Enthält alle Detektionen eines einzelnen Frames.

    Attributes:
        frame_idx:   0-basierter Frame-Index im Video.
        boxes:       Liste von [x1, y1, x2, y2] in Pixel-Koordinaten (float32).
        confidences: Confidence-Wert pro Box (0.0 – 1.0).
        class_ids:   Klassen-ID pro Box (siehe CLASS_* Konstanten).
        class_names: Klassen-Name pro Box als String.
    """

    frame_idx: int
    boxes: List[np.ndarray] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.boxes)

    def center(self, i: int) -> Tuple[float, float]:
        """Mittelpunkt der i-ten Bounding-Box."""
        b = self.boxes[i]
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def filter_by_class(self, *class_ids: int) -> "DetectionResult":
        """Gibt eine neue DetectionResult zurück, die nur die gewünschten Klassen enthält."""
        allowed = set(class_ids)
        indices = [i for i, c in enumerate(self.class_ids) if c in allowed]
        return DetectionResult(
            frame_idx=self.frame_idx,
            boxes=[self.boxes[i] for i in indices],
            confidences=[self.confidences[i] for i in indices],
            class_ids=[self.class_ids[i] for i in indices],
            class_names=[self.class_names[i] for i in indices],
        )

    def players_only(self) -> "DetectionResult":
        """Filtert auf Klasse player (0) — MVP-Standard."""
        return self.filter_by_class(CLASS_PLAYER)
