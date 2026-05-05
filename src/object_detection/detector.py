"""YOLOv11-based object detector for UniVision2Board (Phase 3).

Detects players, goalkeepers and optionally balls per video frame.
Uses the pretrained ultralytics YOLOv11 model.

Domain mapping from COCO classes:
  - person (0)       → player (default) or goalkeeper (override via goalkeeper_ids)
  - sports ball (32) → ball  (optional, disabled by default)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
from ultralytics import YOLO

from .detection import (
    COCO_PERSON,
    COCO_SPORTS_BALL,
    DetectionResult,
)

logger = logging.getLogger(__name__)
_DEBUG_LOG_PATH = Path("/home/admin/KIP/.cursor/debug-4e2d6a.log")
_DEBUG_SESSION_ID = "4e2d6a"

# Default confidence thresholds per domain class
_DEFAULT_CONF: dict[str, float] = {
    "player":     0.40,
    "goalkeeper": 0.40,
    "ball":       0.25,
}

# IoU-Schwelle für YOLO-interne NMS (niedriger = aggressiver, weniger Duplikate)
_NMS_IOU_THRESHOLD: float = 0.40

# Zusätzlicher Post-NMS IoU-Grenzwert zum Entfernen überlappender Player-Boxen
_POST_NMS_IOU: float = 0.45

# Größenfilter: Spieler-Bounding-Box darf maximal X% des Frames überspannen.
# Verhindert, dass Tore/Banden/Anzeigetafeln als Spieler erkannt werden.
_MAX_BOX_WIDTH_FRAC:  float = 0.15   # max 15% der Frame-Breite
_MAX_BOX_HEIGHT_FRAC: float = 0.50   # max 50% der Frame-Höhe
_MIN_BOX_WIDTH_FRAC:  float = 0.003  # min 0.3% — weit entfernte Spieler können sehr klein sein
_MIN_BOX_HEIGHT_FRAC: float = 0.007  # min 0.7%
# Aspektverhältnis h/w eines stehenden Spielers.
# Weit entfernte Spieler erscheinen kleiner und kompakter → tieferes Minimum.
# Falsch erkannte Strukturen (Tore, Banden) tendieren zu <1.2 (sehr breit).
_MIN_ASPECT_RATIO: float = 1.2
_MAX_ASPECT_RATIO: float = 8.0

# Default YOLOv11 nano weights (auto-downloaded by ultralytics on first use)
DEFAULT_MODEL = "yolo11n.pt"


class Detector:
    """Wraps a YOLOv11 model and maps detections to domain classes.

    Args:
        model_path:         Path to .pt weights file, or model name for
                            automatic download (e.g. 'yolo11n.pt').
        conf_thresholds:    Dict mapping class name → min confidence.
                            Defaults to built-in thresholds per class.
        detect_ball:        Whether to include ball detections.
        goalkeeper_ids:     Set of track/detection IDs that should be
                            labelled as 'goalkeeper' instead of 'player'.
                            Used for manual overrides from Phase 5 onwards.
        device:             Inference device ('cpu', 'cuda', '0', …).
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        conf_thresholds: dict[str, float] | None = None,
        detect_ball: bool = False,
        goalkeeper_ids: set[int] | None = None,
        device: str = "cpu",
    ) -> None:
        self._debug_run_id = f"detector_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        self._model_path = str(model_path)
        self._detect_ball = detect_ball
        self._goalkeeper_ids: set[int] = goalkeeper_ids or set()
        self._device = device

        self.conf_thresholds: dict[str, float] = {**_DEFAULT_CONF}
        if conf_thresholds:
            self.conf_thresholds.update(conf_thresholds)

        logger.info("Lade YOLO-Modell: %s (device=%s)", self._model_path, device)
        self._model = YOLO(self._model_path)

        # Detect whether the model already uses domain class names directly.
        _domain_names = {"player", "goalkeeper", "ball", "referee"}
        model_names = set(self._model.names.values())
        self._uses_domain_classes: bool = bool(model_names & _domain_names)
        if self._uses_domain_classes:
            self._domain_class_map: dict[int, str] = {
                k: v for k, v in self._model.names.items() if v in _domain_names
            }
            logger.info("Fine-Tuned Modell erkannt – Domain-Klassen direkt gemappt: %s", self._domain_class_map)
        else:
            self._domain_class_map = {}

        logger.info("Modell geladen.")

    def _debug_log(self, hypothesis_id: str, location: str, message: str, data: dict) -> None:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": self._debug_run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        try:
            with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = -1,
    ) -> list[DetectionResult]:
        """Run detection on a single BGR frame.

        Args:
            frame_bgr:   OpenCV BGR image as NumPy array.
            frame_index: Optional frame index for bookkeeping.

        Returns:
            List of DetectionResult objects passing confidence thresholds.
        """
        # Run YOLO mit niedrigem Global-Threshold; per-Klassen-Filter danach.
        # iou=_NMS_IOU_THRESHOLD: aggressivere YOLO-interne NMS verhindert
        # doppelte Boxen für denselben Spieler schon im Modell-Output.
        results = self._model(
            frame_bgr,
            verbose=False,
            device=self._device,
            conf=min(self.conf_thresholds.values()),
            iou=_NMS_IOU_THRESHOLD,
        )

        frame_h, frame_w = frame_bgr.shape[:2]

        detections: list[DetectionResult] = []
        raw_person_conf: list[float] = []
        raw_person_count = 0
        filtered_low_conf_count = 0
        filtered_size_count = 0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])

                if self._uses_domain_classes:
                    domain = self._domain_class_map.get(cls_id)
                    if domain == "player":
                        raw_person_count += 1
                        raw_person_conf.append(conf)
                else:
                    if cls_id == COCO_PERSON:
                        raw_person_count += 1
                        raw_person_conf.append(conf)
                    domain = self._map_class(cls_id)

                if domain is None:
                    continue
                if domain == "ball" and not self._detect_ball:
                    continue

                # ── Größen- und Aspektverhältnis-Filter (alle Personen-Klassen) ──
                # "referee" hat oft riesige Fehlboxen → gleiche Filterung.
                # Bälle sind rund/klein – kein Aspektverhältnis-Check nötig.
                if domain in ("player", "goalkeeper", "referee"):
                    box_w = x2 - x1
                    box_h = y2 - y1
                    rel_w = box_w / frame_w
                    rel_h = box_h / frame_h
                    aspect = box_h / box_w if box_w > 0 else 0.0
                    if (
                        rel_w > _MAX_BOX_WIDTH_FRAC
                        or rel_h > _MAX_BOX_HEIGHT_FRAC
                        or rel_w < _MIN_BOX_WIDTH_FRAC
                        or rel_h < _MIN_BOX_HEIGHT_FRAC
                        or aspect < _MIN_ASPECT_RATIO
                        or aspect > _MAX_ASPECT_RATIO
                    ):
                        filtered_size_count += 1
                        continue

                if conf < self.conf_thresholds.get(domain, 0.0):
                    filtered_low_conf_count += 1
                    continue

                detections.append(
                    DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        class_id=DetectionResult.CLASS_PLAYER
                        if domain == "player"
                        else DetectionResult.CLASS_GOALKEEPER
                        if domain == "goalkeeper"
                        else DetectionResult.CLASS_BALL,
                        class_name=domain,
                        confidence=conf,
                        frame_index=frame_index,
                    )
                )

        if frame_index >= 0 and (frame_index < 5 or frame_index % 30 == 0):
            conf_min = min(raw_person_conf) if raw_person_conf else None
            conf_max = max(raw_person_conf) if raw_person_conf else None
            # region agent log
            self._debug_log(
                "H1",
                "src/object_detection/detector.py:detect",
                "detection_filter_stats",
                {
                    "frame_index": frame_index,
                    "model": self._model_path,
                    "threshold_player": self.conf_thresholds.get("player"),
                    "raw_person_count": raw_person_count,
                    "filtered_low_conf_count": filtered_low_conf_count,
                    "filtered_size_count": filtered_size_count,
                    "kept_detection_count": len(detections),
                    "raw_person_conf_min": conf_min,
                    "raw_person_conf_max": conf_max,
                },
            )
            # endregion

        # Post-NMS: überlappende Spieler-Boxen klassen-übergreifend entfernen
        before_nms = len(detections)
        detections = self._apply_post_nms(detections)
        removed = before_nms - len(detections)

        logger.debug(
            "Frame %d: %d Detektionen (%d Spieler, %d TH, %d Ball) – Post-NMS: -%d Duplikate",
            frame_index,
            len(detections),
            sum(1 for d in detections if d.class_name == "player"),
            sum(1 for d in detections if d.class_name == "goalkeeper"),
            sum(1 for d in detections if d.class_name == "ball"),
            removed,
        )
        return detections

    def detect_sequence(
        self,
        frames: Sequence[np.ndarray],
        start_index: int = 0,
    ) -> list[list[DetectionResult]]:
        """Run detection on a sequence of frames.

        Returns a list of detection lists, one per frame.
        """
        return [
            self.detect(frame, frame_index=start_index + i)
            for i, frame in enumerate(frames)
        ]

    def set_goalkeeper(self, detection_idx: int) -> None:
        """Mark a detection index as goalkeeper for subsequent frames."""
        self._goalkeeper_ids.add(detection_idx)

    def set_conf_threshold(self, class_name: str, threshold: float) -> None:
        """Update the confidence threshold for a specific class at runtime."""
        if class_name not in _DEFAULT_CONF:
            raise ValueError(
                f"Unbekannte Klasse '{class_name}'. "
                f"Gueltig: {list(_DEFAULT_CONF)}"
            )
        self.conf_thresholds[class_name] = threshold
        logger.info("Conf-Schwelle fuer '%s' auf %.2f gesetzt.", class_name, threshold)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_post_nms(
        detections: list[DetectionResult],
        iou_threshold: float = _POST_NMS_IOU,
    ) -> list[DetectionResult]:
        """Entfernt überlappende Detektionen per greedy IoU-NMS.

        Sortiert nach Konfidenz (absteigend) und verwirft jede Detection,
        deren IoU mit einer bereits behaltenen Box den Schwellwert überschreitet.
        Verhindert Mehrfacherkennungen desselben Spielers.
        """
        if len(detections) <= 1:
            return detections

        # Absteigend nach Konfidenz sortieren
        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[DetectionResult] = []

        for candidate in dets:
            cx1, cy1, cx2, cy2 = candidate.bbox
            suppress = False
            for ref in kept:
                rx1, ry1, rx2, ry2 = ref.bbox
                # Intersection
                ix1 = max(cx1, rx1)
                iy1 = max(cy1, ry1)
                ix2 = min(cx2, rx2)
                iy2 = min(cy2, ry2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter == 0.0:
                    continue
                area_c = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
                area_r = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
                union = area_c + area_r - inter
                iou = inter / union if union > 0 else 0.0
                if iou > iou_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(candidate)

        return kept

    def _map_class(self, coco_cls: int) -> str | None:
        """Map a COCO class ID to a domain class name, or None to skip."""
        if coco_cls == COCO_PERSON:
            return "player"
        if coco_cls == COCO_SPORTS_BALL and self._detect_ball:
            return "ball"
        return None
