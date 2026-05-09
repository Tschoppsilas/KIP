"""Ausgabeformat für einen Tracking-Durchlauf pro Frame."""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class TrackResult:
    """
    Enthält alle getrackten Spieler eines einzelnen Frames.

    Attributes:
        frame_idx:   0-basierter Frame-Index im Video.
        track_ids:   Konsistente Spieler-ID über Frames hinweg.
        boxes:       Liste von [x1, y1, x2, y2] in Pixel-Koordinaten (float32).
        confidences: Confidence-Wert pro Box (0.0 – 1.0).
    """

    frame_idx: int
    track_ids: List[int] = field(default_factory=list)
    boxes: List[np.ndarray] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.track_ids)

    def center(self, i: int) -> Tuple[float, float]:
        """Mittelpunkt der i-ten Bounding-Box."""
        b = self.boxes[i]
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def centers(self) -> List[Tuple[float, float]]:
        """Mittelpunkte aller getrackten Boxes."""
        return [self.center(i) for i in range(len(self))]

    def filtered(self, excluded_ids) -> "TrackResult":
        """Gibt eine Kopie ohne die angegebenen Track-IDs zurück."""
        idx = [i for i, tid in enumerate(self.track_ids) if tid not in excluded_ids]
        return TrackResult(
            frame_idx=self.frame_idx,
            track_ids=[self.track_ids[i] for i in idx],
            boxes=[self.boxes[i] for i in idx],
            confidences=[self.confidences[i] for i in idx],
        )
