"""Homography-Transformation: Videokoordinaten → Taktikboard-Koordinaten."""

import json

import cv2
import numpy as np
from typing import List, Optional, Tuple

from src.utils import get_logger

logger = get_logger(__name__)


class HomographyTransformer:
    """Berechnet und wendet eine perspektivische Homography an."""

    def __init__(self):
        self.matrix: Optional[np.ndarray] = None
        self.src_points: List[Tuple[float, float]] = []
        self.dst_points: List[Tuple[float, float]] = []

    def compute(
        self,
        src_points: List[Tuple[float, float]],
        dst_points: List[Tuple[float, float]],
    ) -> "HomographyTransformer":
        """Berechnet die Homography-Matrix aus mindestens 4 Punkt-Paaren."""
        if len(src_points) < 4 or len(dst_points) < 4:
            raise ValueError("Mindestens 4 Punkt-Paare erforderlich.")
        if len(src_points) != len(dst_points):
            raise ValueError("Anzahl der Quell- und Zielpunkte muss übereinstimmen.")

        self.src_points = list(src_points)
        self.dst_points = list(dst_points)

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)

        self.matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if self.matrix is None:
            raise RuntimeError("Homography konnte nicht berechnet werden (zu wenige/kollineare Punkte?).")

        logger.info(f"Homography berechnet aus {len(src_points)} Punkten.")
        return self

    def transform(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transformiert eine Liste von Video-Koordinaten zu Board-Koordinaten."""
        if self.matrix is None:
            raise RuntimeError("Homography noch nicht berechnet – zuerst compute() aufrufen.")
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.matrix)
        return [(float(p[0][0]), float(p[0][1])) for p in transformed]

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transformiert einen einzelnen Punkt."""
        return self.transform([(x, y)])[0]

    def save(self, path: str) -> None:
        """Speichert Kalibrierungspunkte als JSON (keine Matrix, da reproduzierbar)."""
        data = {
            "src_points": self.src_points,
            "dst_points": self.dst_points,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Kalibrierung gespeichert: {path}")

    @classmethod
    def load(cls, path: str) -> "HomographyTransformer":
        """Lädt Kalibrierungspunkte aus JSON und berechnet die Homography."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        transformer = cls()
        transformer.compute(
            [tuple(p) for p in data["src_points"]],
            [tuple(p) for p in data["dst_points"]],
        )
        logger.info(f"Kalibrierung geladen: {path}")
        return transformer
