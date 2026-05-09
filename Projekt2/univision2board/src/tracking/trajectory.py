"""Laufweg-Verwaltung pro Spieler-ID."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from src.video_processing.homography import HomographyTransformer


@dataclass
class Trajectory:
    """
    Speichert den Laufweg eines einzelnen Spielers als geordnete Punktfolge.

    Attributes:
        track_id: ByteTrack-ID des Spielers.
        points:   Liste von (frame_idx, x, y) in Videokoordinaten (Pixel).
    """

    track_id: int
    points: List[Tuple[int, float, float]] = field(default_factory=list)

    def add_point(self, frame_idx: int, x: float, y: float) -> None:
        """Fügt einen neuen Raum-Zeitpunkt zum Laufweg hinzu."""
        self.points.append((frame_idx, x, y))

    def xy_sequence(self) -> List[Tuple[float, float]]:
        """Gibt die reinen (x, y)-Koordinaten zurück (ohne frame_idx)."""
        return [(x, y) for _, x, y in self.points]

    def to_board_coords(self, transformer: "HomographyTransformer") -> "Trajectory":
        """
        Gibt eine neue Trajectory zurück, deren Punkte auf
        Taktikboard-Koordinaten transformiert sind.
        """
        transformed = Trajectory(track_id=self.track_id)
        for frame_idx, x, y in self.points:
            bx, by = transformer.transform_point(x, y)
            transformed.points.append((frame_idx, bx, by))
        return transformed

    def last_point(self) -> Optional[Tuple[int, float, float]]:
        """Letzter bekannter Raum-Zeitpunkt des Spielers."""
        return self.points[-1] if self.points else None
