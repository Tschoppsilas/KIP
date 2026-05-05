"""ByteTrack-based player tracker (Phase 4).

Wraps ``supervision.ByteTrack`` to consume ``DetectionResult`` objects from
Phase 3 and emit ``TrackedPlayer`` objects with stable IDs across frames.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from src.object_detection.detection import DetectionResult
from src.tracking.track import TrackedPlayer


class PlayerTracker:
    """Maintains identity of players across frames using ByteTrack.

    Args:
        track_activation_threshold: Minimum confidence to initialise a new
            track. Lower = more tracks started (default 0.15).
        lost_track_buffer: Frames to keep a lost track alive before dropping it.
            Higher = more stable IDs through occlusions (default 120).
        minimum_matching_threshold: IoU threshold for track-detection matching.
            Lower = more forgiving, fewer ID switches (default 0.45).
        frame_rate: Expected video frame-rate (influences buffer timing).
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.15,
        lost_track_buffer: int = 120,
        minimum_matching_threshold: float = 0.45,
        frame_rate: int = 30,
    ) -> None:
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list[DetectionResult],
        frame_index: int = -1,
    ) -> list[TrackedPlayer]:
        """Process detections for one frame and return tracked players.

        Args:
            detections:  List of ``DetectionResult`` objects from Phase 3.
            frame_index: Frame index (used for trajectory logging).

        Returns:
            List of ``TrackedPlayer`` objects with stable ``track_id``.
        """
        sv_dets = self._to_sv_detections(detections)
        tracked = self._tracker.update_with_detections(sv_dets)
        return self._from_sv_detections(tracked, detections, frame_index)

    def reset(self) -> None:
        """Reset tracker state (call between independent video sequences)."""
        self._tracker.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_sv_detections(detections: list[DetectionResult]) -> sv.Detections:
        """Convert DetectionResult list to a supervision Detections object."""
        if not detections:
            return sv.Detections.empty()

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    @staticmethod
    def _from_sv_detections(
        sv_dets: sv.Detections,
        original: list[DetectionResult],
        frame_index: int,
    ) -> list[TrackedPlayer]:
        """Convert a tracked supervision Detections back to TrackedPlayer list.

        ``sv.ByteTrack.update_with_detections`` returns a filtered Detections
        object that only contains successfully tracked boxes, with
        ``tracker_id`` populated.  We match them back by bounding-box index
        using the order preserved by supervision.
        """
        if sv_dets.tracker_id is None or len(sv_dets) == 0:
            return []

        # Build a lookup from bbox (as rounded tuple) → class_name from originals
        bbox_to_class: dict[tuple, str] = {
            tuple(np.round(d.bbox, 1)): d.class_name for d in original
        }

        players: list[TrackedPlayer] = []
        for i, (bbox, conf, tid) in enumerate(
            zip(sv_dets.xyxy, sv_dets.confidence, sv_dets.tracker_id)
        ):
            key = tuple(np.round(bbox, 1))
            class_name = bbox_to_class.get(key, "player")
            players.append(
                TrackedPlayer(
                    track_id=int(tid),
                    bbox=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
                    class_name=class_name,
                    confidence=float(conf),
                    frame_index=frame_index,
                )
            )
        return players
