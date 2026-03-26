"""Tracking data structures: TrackedPlayer and Trajectory (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import numpy as np


@dataclass
class TrackedPlayer:
    """Represents a tracked player over one or more frames.

    Attributes:
        track_id:    Stable ID assigned by ByteTrack across frames.
        bbox:        Current bounding box (x1, y1, x2, y2) in pixel coords.
        class_name:  Domain class label ('player', 'goalkeeper', 'ball').
        confidence:  Detection confidence of the current frame's detection.
        frame_index: Frame index of this observation.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
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


@dataclass
class Trajectory:
    """Ordered sequence of (cx, cy) center positions for one player.

    Optionally holds board-coordinate versions when a homography is applied.

    Attributes:
        track_id:        The ByteTrack ID this trajectory belongs to.
        class_name:      Domain class label of this player.
        points_px:       Center positions in pixel coordinates, ordered by frame.
        points_board:    Center positions in board coordinates (metres),
                         populated after homography transform.
        frame_indices:   Frame index for each point in points_px.
    """

    track_id: int
    class_name: str = "player"
    points_px: list[tuple[float, float]] = field(default_factory=list)
    points_board: list[tuple[float, float]] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)

    def add_point(
        self,
        center_px: tuple[float, float],
        frame_index: int = -1,
        center_board: tuple[float, float] | None = None,
    ) -> None:
        """Append a new position to the trajectory."""
        self.points_px.append(center_px)
        self.frame_indices.append(frame_index)
        if center_board is not None:
            self.points_board.append(center_board)

    def smooth(self, window: int = 3) -> list[tuple[float, float]]:
        """Return a smoothed version of points_px using a moving average.

        Args:
            window: Number of consecutive points to average (odd recommended).

        Returns:
            Smoothed list of (cx, cy) positions, same length as points_px.
        """
        if len(self.points_px) < 2:
            return list(self.points_px)

        arr = np.array(self.points_px, dtype=np.float64)
        half = window // 2
        smoothed = []
        for i in range(len(arr)):
            lo = max(0, i - half)
            hi = min(len(arr), i + half + 1)
            smoothed.append(tuple(arr[lo:hi].mean(axis=0).tolist()))
        return smoothed  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self.points_px)


def build_trajectories(
    tracked_frames: list[list[TrackedPlayer]],
) -> dict[int, Trajectory]:
    """Build a Trajectory per track_id from a list of per-frame player lists.

    Args:
        tracked_frames: List of frames; each frame is a list of TrackedPlayer.

    Returns:
        Dict mapping track_id → Trajectory.
    """
    trajectories: dict[int, Trajectory] = {}
    for players in tracked_frames:
        for player in players:
            if player.track_id not in trajectories:
                trajectories[player.track_id] = Trajectory(
                    track_id=player.track_id,
                    class_name=player.class_name,
                )
            trajectories[player.track_id].add_point(
                player.center,
                frame_index=player.frame_index,
            )
    return trajectories
