"""Roboflow-Datensatz herunterladen und in YOLO-Format für UniVision2Board aufbereiten.

Aufruf:
    python finetune/download_dataset.py --api-key DEIN_KEY --workspace WORKSPACE --project PROJECT --version VERSION

Beispiel (hockey-Datensatz von Roboflow):
    python finetune/download_dataset.py \\
        --api-key abc123 \\
        --workspace sportcontract \\
        --project hockey-fwm0b \\
        --version 1

Nach dem Download:
    finetune/dataset/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
    finetune/data.yaml  (wird automatisch aktualisiert)

Danach: python finetune/train.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR  = PROJECT_ROOT / "finetune" / "dataset"
DATA_YAML    = PROJECT_ROOT / "finetune" / "data.yaml"

# Klassen-Mapping: Roboflow-Klassen → UniVision2Board-Klassen
# Anpassen falls der gewählte Datensatz andere Klassennamen verwendet.
CLASS_MAP = {
    # Spieler
    "player":     0,
    "players":    0,
    "Player":     0,
    "human":      0,
    "person":     0,
    # Torwart
    "goalkeeper": 1,
    "goalie":     1,
    "Goalkeeper": 1,
    # Ball
    "ball":       2,
    "Ball":       2,
    "puck":       2,
    "Puck":       2,
}


def download_and_remap(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
) -> Path:
    """Lädt einen Roboflow-Datensatz herunter und mappt Klassen auf UniVision-Schema.

    Returns:
        Pfad zum fertigen Dataset-Verzeichnis.
    """
    from roboflow import Roboflow

    print(f"Verbinde mit Roboflow: {workspace}/{project} v{version} …")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov11", location=str(DATASET_DIR / "_raw"))

    raw_dir = Path(dataset.location)
    print(f"Heruntergeladen nach: {raw_dir}")

    # Klassen aus der heruntergeladenen data.yaml lesen
    import yaml
    raw_yaml = raw_dir / "data.yaml"
    if raw_yaml.exists():
        with open(raw_yaml) as f:
            raw_cfg = yaml.safe_load(f)
        raw_names: list[str] = raw_cfg.get("names", [])
        print(f"Roboflow-Klassen: {raw_names}")
    else:
        raw_names = []
        print("[WARN] Keine data.yaml im heruntergeladenen Datensatz.")

    # Bilder + Labels kopieren und Klassen remappen
    for split in ("train", "valid", "val", "test"):
        src_img = raw_dir / split / "images"
        src_lbl = raw_dir / split / "labels"
        if not src_img.exists():
            continue

        dst_split = "val" if split in ("valid", "val") else split
        dst_img = DATASET_DIR / "images" / dst_split
        dst_lbl = DATASET_DIR / "labels" / dst_split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for img_path in src_img.iterdir():
            shutil.copy2(img_path, dst_img / img_path.name)

        n_kept = 0
        for lbl_path in src_lbl.iterdir():
            lines_out = []
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if not parts:
                    continue
                raw_cid = int(parts[0])
                raw_name = raw_names[raw_cid] if raw_cid < len(raw_names) else ""
                mapped = CLASS_MAP.get(raw_name, CLASS_MAP.get(str(raw_cid)))
                if mapped is None:
                    continue   # Klasse nicht relevant (Referee, Schiedsrichter, …)
                lines_out.append(f"{mapped} {' '.join(parts[1:])}")
                n_kept += 1
            (dst_lbl / lbl_path.name).write_text("\n".join(lines_out))

        print(f"  {dst_split}: {len(list(dst_img.iterdir()))} Bilder, {n_kept} Labels übernommen")

    print(f"\nDataset bereit: {DATASET_DIR}")
    print("Nächster Schritt: python finetune/train.py")
    return DATASET_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",   required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project",   required=True)
    parser.add_argument("--version",   type=int, default=1)
    args = parser.parse_args()

    download_and_remap(args.api_key, args.workspace, args.project, args.version)
