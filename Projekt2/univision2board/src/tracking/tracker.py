"""
Spieler-Tracker auf Basis von ByteTrack (via ultralytics model.track()).

ByteTrack verknüpft die per YOLOv11 erkannten Bounding-Boxes über Frames
hinweg und liefert konsistente Track-IDs, auch bei kurzem Verdecken
eines Spielers (IoU-basiertes Re-Matching über einen Puffer).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from src.tracking.track_result import TrackResult
from src.tracking.trajectory import Trajectory
from src.utils import get_logger

logger = get_logger(__name__)

_DEFAULT_TARGET_CLASSES = [0]  # nur player


class PlayerTracker:
    """
    Verbindet YOLO-Detection und ByteTrack-Tracking in einem Schritt.

    Verwendet `model.track(..., tracker="bytetrack.yaml", persist=True)`,
    sodass ByteTrack den internen Zustand frame-übergreifend hält.

    Parameters:
        model_path:      Pfad zu best.pt.
        conf:            Confidence-Schwelle für Detection.
        target_classes:  Klassen-IDs, die als Spieler gewertet werden.
        roi:             Optionales ROI-Rechteck [x1, y1, x2, y2].
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.35,
        target_classes: Optional[List[int]] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ):
        self._model = YOLO(model_path)
        self.conf = conf
        self.target_classes = target_classes if target_classes is not None else _DEFAULT_TARGET_CLASSES
        self.roi = roi
        self._trajectories: Dict[int, Trajectory] = {}
        logger.info(
            f"PlayerTracker bereit: {model_path} | "
            f"conf={conf} | classes={self.target_classes} | roi={roi}"
        )

    # ------------------------------------------------------------------
    # Haupt-API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, frame_idx: int = 0) -> TrackResult:
        """
        Verarbeitet einen einzelnen Frame:
        führt Detection + ByteTrack durch und aktualisiert Laufwege.

        Returns:
            TrackResult mit stabilen IDs für diesen Frame.
        """
        results = self._model.track(
            frame,
            conf=self.conf,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
        track = self._parse(results[0], frame_idx)

        for i in range(len(track)):
            tid = track.track_ids[i]
            cx, cy = track.center(i)
            if tid not in self._trajectories:
                self._trajectories[tid] = Trajectory(track_id=tid)
            self._trajectories[tid].add_point(frame_idx, cx, cy)

        return track

    def get_trajectory(self, track_id: int) -> Optional[Trajectory]:
        """Gibt den Laufweg eines Spielers zurück (None wenn unbekannt)."""
        return self._trajectories.get(track_id)

    def get_all_trajectories(self) -> Dict[int, Trajectory]:
        """Gibt alle bisher gesammelten Laufwege zurück."""
        return dict(self._trajectories)

    def reset(self) -> None:
        """Setzt Tracking-Zustand und Laufwege komplett zurück."""
        self._trajectories.clear()
        # ByteTrack-State im ultralytics-Predictor zurücksetzen
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            self._model.predictor = None
        logger.info("PlayerTracker zurückgesetzt.")

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------

    def _parse(self, result, frame_idx: int) -> TrackResult:
        track = TrackResult(frame_idx=frame_idx)

        if result.boxes is None or result.boxes.id is None:
            return track

        xyxy = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for box, tid, conf, cls_id in zip(xyxy, ids, confs, clss):
            if cls_id not in self.target_classes:
                continue
            if self.roi is not None and not self._in_roi(box):
                continue
            track.track_ids.append(int(tid))
            track.boxes.append(box.astype(np.float32))
            track.confidences.append(float(conf))

        return track

    def _in_roi(self, box: np.ndarray) -> bool:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        x1, y1, x2, y2 = self.roi
        return x1 <= cx <= x2 and y1 <= cy <= y2
