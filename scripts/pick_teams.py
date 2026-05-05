"""Interaktiver Team-Farb-Picker für UniVision2Board.

Zeigt das erste Frame des Videos mit YOLO-Detektionen an.
Der User klickt auf einen Spieler von Team A und einen von Team B.
Die HSV-Referenzfarben werden in 'team_colors.json' gespeichert und
von visualize_pipeline.py automatisch geladen.

Bedienung:
  - Linksklick auf einen Spieler → wählt die Bounding-Box aus (gelb markiert)
  - Taste 'a' → weist die gewählte Box Team A zu (rot)
  - Taste 'b' → weist die gewählte Box Team B zu (blau)
  - Taste 's' / Enter → speichert und beendet (nur wenn beide Teams gewählt)
  - Taste 'r' → Auswahl zurücksetzen
  - Taste 'q' / Esc → Abbrechen ohne Speichern

Aufruf:
    python scripts/pick_teams.py [VIDEO] [OUTPUT_JSON]

Beispiele:
    python scripts/pick_teams.py Videos/Muenchenstein_1.mp4
    python scripts/pick_teams.py Videos/Muenchenstein_1.mp4 team_colors.json
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.object_detection.detector import Detector
from src.tracking.team_assigner import extract_hsv_feature, TEAM_A, TEAM_B

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("pick_teams")

# ---------------------------------------------------------------------------
# Argumente
# ---------------------------------------------------------------------------
VIDEO      = sys.argv[1] if len(sys.argv) > 1 else "Videos/Muenchenstein_1.mp4"
OUTPUT     = sys.argv[2] if len(sys.argv) > 2 else "team_colors.json"
_MODEL     = "finetune/runs/train/weights/best.pt"
MODEL      = _MODEL if Path(_MODEL).exists() else "yolo11n.pt"

# ---------------------------------------------------------------------------
# Farben (BGR)
# ---------------------------------------------------------------------------
C_DEFAULT  = (180, 180, 180)   # Grau: erkannt, nicht gewählt
C_SELECTED = (0, 220, 220)     # Gelb: aktuell angeklickt
C_TEAM_A   = (0,  50, 220)     # Rot: Team A
C_TEAM_B   = (220, 50,   0)    # Blau: Team B
C_TEXT     = (255, 255, 255)

WINDOW = "Team-Farb-Picker — Klick auf Spieler, dann A / B drücken"


def _put_label(img, text, x, y, color):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)


def main() -> None:
    # --- Erstes Frame laden ---
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        logger.error("Video konnte nicht geöffnet werden: %s", VIDEO)
        sys.exit(1)
    ok, frame_orig = cap.read()
    cap.release()
    if not ok:
        logger.error("Erstes Frame konnte nicht gelesen werden.")
        sys.exit(1)

    # --- YOLO Detection auf erstem Frame ---
    logger.info("Lade YOLO-Modell: %s", MODEL)
    detector = Detector(MODEL, conf_thresholds={"player": 0.18, "goalkeeper": 0.18},
                        detect_ball=False)
    dets = [d for d in detector.detect(frame_orig, frame_index=0)
            if d.class_name in ("player", "goalkeeper")]
    logger.info("%d Spieler erkannt.", len(dets))

    if len(dets) < 2:
        logger.error("Zu wenige Spieler erkannt (min. 2 benötigt).")
        sys.exit(1)

    bboxes = [d.bbox for d in dets]

    # --- Zustand ---
    selected_idx: int | None = None
    team_a_idx:   int | None = None
    team_b_idx:   int | None = None

    def _draw(frame_base: np.ndarray) -> np.ndarray:
        img = frame_base.copy()
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = (int(v) for v in bbox)
            if i == team_a_idx:
                color, label = C_TEAM_A, "TEAM A"
            elif i == team_b_idx:
                color, label = C_TEAM_B, "TEAM B"
            elif i == selected_idx:
                color, label = C_SELECTED, "< gewählt"
            else:
                color, label = C_DEFAULT, ""
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            if label:
                _put_label(img, label, x1, y1 - 6, color)

        # Statuszeile unten
        status_parts = []
        if team_a_idx is None:
            status_parts.append("Team A: nicht gewählt")
        else:
            status_parts.append("Team A: OK ✓")
        if team_b_idx is None:
            status_parts.append("Team B: nicht gewählt")
        else:
            status_parts.append("Team B: OK ✓")

        both_ok = team_a_idx is not None and team_b_idx is not None
        hint = "  |  [S/Enter] Speichern" if both_ok else ""
        status = "  |  ".join(status_parts) + hint + "  |  [R] Reset  [Q] Abbrechen"

        h = img.shape[0]
        cv2.rectangle(img, (0, h - 36), (img.shape[1], h), (20, 20, 20), -1)
        cv2.putText(img, status, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, C_TEXT, 1, cv2.LINE_AA)

        # Legende oben
        cv2.rectangle(img, (0, 0), (460, 38), (20, 20, 20), -1)
        legend = "Klick = Box wählen  |  A = Team A  |  B = Team B"
        cv2.putText(img, legend, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, C_TEXT, 1, cv2.LINE_AA)
        return img

    def _on_mouse(event, x, y, flags, param):
        nonlocal selected_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Nächste Box zum Klickpunkt finden
        best_i, best_d = None, float("inf")
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d = (x - cx) ** 2 + (y - cy) ** 2
                if d < best_d:
                    best_d, best_i = d, i
        selected_idx = best_i
        cv2.imshow(WINDOW, _draw(frame_orig))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1280, frame_orig.shape[1]),
                     min(720,  frame_orig.shape[0]))
    cv2.setMouseCallback(WINDOW, _on_mouse)
    cv2.imshow(WINDOW, _draw(frame_orig))

    logger.info("Fenster geöffnet. Klicke auf Spieler, dann A / B drücken.")

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):   # Q / Esc
            logger.info("Abgebrochen ohne Speichern.")
            cv2.destroyAllWindows()
            sys.exit(0)

        elif key == ord('r'):       # Reset
            selected_idx = team_a_idx = team_b_idx = None
            cv2.imshow(WINDOW, _draw(frame_orig))

        elif key == ord('a') and selected_idx is not None:
            team_a_idx = selected_idx
            selected_idx = None
            cv2.imshow(WINDOW, _draw(frame_orig))
            logger.info("Team A gesetzt auf Box %d.", team_a_idx)

        elif key == ord('b') and selected_idx is not None:
            team_b_idx = selected_idx
            selected_idx = None
            cv2.imshow(WINDOW, _draw(frame_orig))
            logger.info("Team B gesetzt auf Box %d.", team_b_idx)

        elif key in (ord('s'), 13):  # S / Enter
            if team_a_idx is None or team_b_idx is None:
                logger.warning("Bitte erst beide Teams wählen (A und B).")
                continue

            feat_a = extract_hsv_feature(frame_orig, bboxes[team_a_idx])
            feat_b = extract_hsv_feature(frame_orig, bboxes[team_b_idx])
            if feat_a is None or feat_b is None:
                logger.error("HSV-Feature konnte nicht extrahiert werden.")
                continue

            data = {
                "team_a_hsv": feat_a.tolist(),
                "team_b_hsv": feat_b.tolist(),
                "bbox_a": list(bboxes[team_a_idx]),
                "bbox_b": list(bboxes[team_b_idx]),
            }
            out = Path(OUTPUT)
            out.write_text(json.dumps(data, indent=2))
            logger.info("Gespeichert: %s", out.resolve())
            logger.info("  Team A HSV: %s", [f"{v:.1f}" for v in feat_a])
            logger.info("  Team B HSV: %s", [f"{v:.1f}" for v in feat_b])
            cv2.destroyAllWindows()
            break

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            logger.info("Fenster geschlossen.")
            break


if __name__ == "__main__":
    main()
