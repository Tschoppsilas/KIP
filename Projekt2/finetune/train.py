"""Fine-Tuning Script: YOLOv11 auf Unihockey-Datensatz.

Trainiert auf dem vorbereiteten Datensatz (finetune/dataset/) und speichert
das beste Modell unter finetune/runs/train/best.pt.

Aufruf:
    python finetune/train.py [--epochs 30] [--base yolo11n.pt] [--device cpu]

Empfehlung:
    - CPU:  --epochs 20 --base yolo11n.pt   (langsam, aber funktional)
    - GPU:  --epochs 50 --base yolo11s.pt   (schneller, besser)
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_YAML    = PROJECT_ROOT / "finetune" / "data.yaml"
RUNS_DIR     = PROJECT_ROOT / "finetune" / "runs"


def train(
    base_model: str = "yolo11n.pt",
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cpu",
    patience: int = 10,
    lr0: float = 0.01,
    lrf: float = 0.01,
    resume: bool = False,
) -> Path:
    """Startet das Fine-Tuning.

    Args:
        base_model: Basisgewichte (yolo11n.pt = klein, yolo11s.pt = genauer).
        epochs:     Trainings-Epochen.
        imgsz:      Eingabegrösse in Pixeln.
        batch:      Batch-Grösse (bei CPU: 4–8 empfohlen).
        device:     'cpu' oder '0' (GPU 0).
        patience:   Early-Stopping Geduld in Epochen.

    Returns:
        Pfad zum besten Modell (best.pt).
    """
    from ultralytics import YOLO

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml nicht gefunden: {DATA_YAML}")

    # Prüfen ob Trainingsbilder vorhanden
    train_img = PROJECT_ROOT / "finetune" / "dataset" / "images" / "train"
    if not train_img.exists() or not any(train_img.iterdir()):
        raise FileNotFoundError(
            f"Keine Trainingsbilder unter {train_img}.\n"
            "Führe zuerst aus:\n"
            "  python finetune/prepare_dataset.py   (Auto-Labels aus Videos)\n"
            "  oder\n"
            "  python finetune/download_dataset.py --api-key KEY ...   (Roboflow)"
        )

    # Beim Resume: last.pt laden und Training fortsetzen
    if resume:
        last_pt = RUNS_DIR / "train" / "weights" / "last.pt"
        if not last_pt.exists():
            raise FileNotFoundError(f"Kein last.pt zum Fortsetzen: {last_pt}")
        print(f"Fortsetzen von: {last_pt}")
        model = YOLO(str(last_pt))
        results = model.train(resume=True)
    else:
        print(f"Basismodell:  {base_model}")
        print(f"Datensatz:    {DATA_YAML}")
        print(f"Epochen:      {epochs}  |  Batch: {batch}  |  Device: {device}")
        print()
        model = YOLO(base_model)
        results = model.train(
            data=str(DATA_YAML),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=patience,
            lr0=lr0,
            lrf=lrf,
            project=str(RUNS_DIR),
            name="train",
            exist_ok=True,
            verbose=True,
        )

    best = RUNS_DIR / "train" / "weights" / "best.pt"
    if best.exists():
        print(f"\nBestes Modell gespeichert: {best}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",    default="yolo11n.pt")
    parser.add_argument("--epochs",  type=int,   default=20)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--patience", type=int,   default=10)
    parser.add_argument("--lr0",      type=float, default=0.01,
                        help="Initiale Lernrate (Standard: 0.01)")
    parser.add_argument("--lrf",      type=float, default=0.01,
                        help="Finale Lernrate als Anteil von lr0 (Standard: 0.01)")
    parser.add_argument("--resume",   action="store_true",
                        help="Training von letztem Checkpoint (last.pt) fortsetzen")
    args = parser.parse_args()

    best = train(
        base_model=args.base,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf,
        resume=args.resume,
    )
    print("\nFertig! Evaluierung starten mit:")
    print(f"  python finetune/evaluate.py --model {best}")
