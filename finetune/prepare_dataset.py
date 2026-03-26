"""Automatische Dataset-Vorbereitung (Fallback ohne externen Datensatz).

Extrahiert Frames aus den Münchenstein-Videos, labelt sie automatisch
mit dem aktuellen YOLO-Modell und speichert sie im YOLO-Format.

Dies ist ein Fallback für den Fall, dass kein externer Datensatz verfügbar ist.
Die automatischen Labels sind eine Basis – manuelle Korrekturen verbessern die Qualität.

Aufruf:
    python finetune/prepare_dataset.py [--videos Videos/*.mp4] [--n 200] [--every 15]
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# Pfade
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR  = PROJECT_ROOT / "finetune" / "dataset"
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
IMAGES_VAL   = DATASET_DIR / "images" / "val"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"
LABELS_VAL   = DATASET_DIR / "labels" / "val"


def _bbox_to_yolo(bbox: tuple, img_w: int, img_h: int, class_id: int) -> str:
    """Konvertiert (x1,y1,x2,y2) → YOLO-Zeile 'class cx cy w h' (normiert)."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    # Clamp auf [0,1]
    cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (cx, cy, w, h))
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def prepare_from_videos(
    video_paths: list[Path],
    model_path: str = "yolo11n.pt",
    n_frames_total: int = 300,
    every_nth: int = 15,
    val_split: float = 0.2,
    conf: float = 0.35,
    seed: int = 42,
) -> None:
    """Extrahiert Frames, labelt sie mit YOLO und speichert im YOLO-Format.

    Args:
        video_paths:    Liste von Video-Dateipfaden.
        model_path:     Pfad zum YOLO-Gewichtsfile.
        n_frames_total: Maximale Gesamtzahl extrahierter Frames.
        every_nth:      Nur jeden n-ten Frame extrahieren.
        val_split:      Anteil für Validierungsset (0–1).
        conf:           Confidence-Schwelle für Auto-Labels.
        seed:           Zufallskern für reproduzierbare Aufteilung.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from ultralytics import YOLO
    from src.object_detection.detection import COCO_PERSON, COCO_SPORTS_BALL

    for d in (IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL):
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    frames_info: list[tuple[np.ndarray, Path]] = []   # (frame, video_path)
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"[WARN] Video nicht lesbar: {vp}")
            continue
        fi = 0
        while len(frames_info) < n_frames_total:
            ret, frame = cap.read()
            if not ret:
                break
            if fi % every_nth == 0:
                frames_info.append((frame.copy(), vp))
            fi += 1
        cap.release()
        if len(frames_info) >= n_frames_total:
            break

    print(f"Extrahiert: {len(frames_info)} Frames aus {len(video_paths)} Videos")

    random.seed(seed)
    random.shuffle(frames_info)
    n_val = max(1, int(len(frames_info) * val_split))
    splits = {"val": frames_info[:n_val], "train": frames_info[n_val:]}

    for split_name, items in splits.items():
        img_dir = DATASET_DIR / "images" / split_name
        lbl_dir = DATASET_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx, (frame, vp) in enumerate(items):
            stem = f"{vp.stem}_{split_name}_{idx:04d}"
            img_path = img_dir / f"{stem}.jpg"
            lbl_path = lbl_dir / f"{stem}.txt"

            cv2.imwrite(str(img_path), frame)

            h, w = frame.shape[:2]
            results = model(frame, conf=conf, verbose=False)[0]
            lines = []
            for box in results.boxes:
                cid = int(box.cls[0])
                # COCO person → player (0), sports ball → ball (2)
                if cid == COCO_PERSON:
                    yolo_class = 0   # player
                elif cid == COCO_SPORTS_BALL:
                    yolo_class = 2   # ball
                else:
                    continue
                bbox = tuple(float(v) for v in box.xyxy[0])
                lines.append(_bbox_to_yolo(bbox, w, h, yolo_class))

            lbl_path.write_text("\n".join(lines))

        print(f"  {split_name:5s}: {len(items):3d} Bilder + Labels → {img_dir}")

    print("\nDataset bereit unter:", DATASET_DIR)
    print("Nächster Schritt: python finetune/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Label Dataset aus Videos")
    parser.add_argument("--model",  default="yolo11n.pt")
    parser.add_argument("--n",      type=int, default=300, help="Max Frames")
    parser.add_argument("--every",  type=int, default=15,  help="Jeden n-ten Frame")
    parser.add_argument("--conf",   type=float, default=0.35)
    args = parser.parse_args()

    videos = sorted((PROJECT_ROOT / "Videos").glob("*.mp4"))
    if not videos:
        print("Keine Videos unter Videos/ gefunden.")
        exit(1)

    print(f"Videos: {[v.name for v in videos]}")
    prepare_from_videos(
        video_paths=videos,
        model_path=args.model,
        n_frames_total=args.n,
        every_nth=args.every,
        conf=args.conf,
    )
