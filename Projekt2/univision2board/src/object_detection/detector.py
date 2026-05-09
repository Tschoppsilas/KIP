"""
YOLOv11-Detektor für Spieler-Erkennung.

Strategie zur Reduktion von Nicht-Spieler-Detektionen (Zuschauer, Bank):
  1. Klassen-Filter: Im MVP wird nur Klasse 0 (player) ausgegeben.
     Das Fine-Tune-Modell kennt zusätzlich goalkeeper(1), ball(2), referee(3);
     diese werden standardmäßig herausgefiltert.
  2. Confidence-Schwelle: Detektionen unterhalb von `conf` werden verworfen
     (Standard: 0.35, höher als YOLO-Default, da das Spielfeld dicht besetzt ist).
  3. ROI (Region of Interest): Optionale Pixel-Maske in Form eines Rechtecks
     [x1, y1, x2, y2]. Alle Bounding-Boxes außerhalb des ROI werden verworfen.
     Damit können Tribüne, Auswechselbank und Bildbereiche außerhalb des Felds
     ausgeblendet werden.
"""

from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from src.object_detection.detection_result import (
    CLASS_NAMES,
    CLASS_PLAYER,
    DetectionResult,
)
from src.utils import get_logger

logger = get_logger(__name__)

# Klassen, die im MVP als "Spieler" gewertet werden
_DEFAULT_TARGET_CLASSES = [CLASS_PLAYER]


class YOLODetector:
    """
    Führt Spieler-Erkennung mit einem YOLOv11-Modell durch.

    Parameters:
        model_path:      Pfad zur .pt-Gewichtsdatei (fine-tuned best.pt).
        conf:            Confidence-Schwellwert (0.0 – 1.0).
        target_classes:  Liste der Klassen-IDs, die ausgegeben werden sollen.
                         Standard: [0] (nur player).
        roi:             Optionales ROI-Rechteck [x1, y1, x2, y2] in Pixel.
                         Detektionen außerhalb werden verworfen.
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
        logger.info(
            f"YOLODetector geladen: {model_path} | "
            f"conf={conf} | classes={self.target_classes} | roi={roi}"
        )

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> DetectionResult:
        """
        Erkennt Spieler in einem einzelnen Frame.

        Returns:
            DetectionResult mit gefilterten Bounding-Boxes, Confidences und Klassen.
        """
        results = self._model(frame, conf=self.conf, verbose=False)
        return self._parse(results[0], frame_idx)

    def detect_batch(
        self, frames: List[np.ndarray], start_idx: int = 0
    ) -> List[DetectionResult]:
        """Erkennt Spieler in einer Liste von Frames (effizienteres Batch-Inferenz)."""
        if not frames:
            return []
        results = self._model(frames, conf=self.conf, verbose=False)
        return [self._parse(r, start_idx + i) for i, r in enumerate(results)]

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------
    def _parse(self, result, frame_idx: int) -> DetectionResult:
        """Wandelt einen YOLO-Result-Eintrag in DetectionResult um."""
        boxes_raw = result.boxes
        detection = DetectionResult(frame_idx=frame_idx)

        if boxes_raw is None or len(boxes_raw) == 0:
            return detection

        xyxy = boxes_raw.xyxy.cpu().numpy()        # [N, 4]
        confs = boxes_raw.conf.cpu().numpy()        # [N]
        clss = boxes_raw.cls.cpu().numpy().astype(int)  # [N]

        for box, conf, cls_id in zip(xyxy, confs, clss):
            if cls_id not in self.target_classes:
                continue
            if self.roi is not None and not self._in_roi(box):
                continue
            detection.boxes.append(box.astype(np.float32))
            detection.confidences.append(float(conf))
            detection.class_ids.append(int(cls_id))
            detection.class_names.append(CLASS_NAMES.get(int(cls_id), "unknown"))

        return detection

    def _in_roi(self, box: np.ndarray) -> bool:
        """Prüft ob der Mittelpunkt der Box innerhalb des ROI liegt."""
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        x1, y1, x2, y2 = self.roi
        return x1 <= cx <= x2 and y1 <= cy <= y2
