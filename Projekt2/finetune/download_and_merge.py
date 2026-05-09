"""Lädt die drei Floorball-Datensätze herunter und führt sie zusammen.

Klassen-Mapping → UniVision2Board-Schema:
  0 = player
  1 = goalkeeper
  2 = ball
"""
from __future__ import annotations
import os, shutil, yaml
from pathlib import Path

API_KEY = os.environ.get("RF_API_KEY", "")
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR  = PROJECT_ROOT / "finetune" / "dataset"
RAW_DIR      = PROJECT_ROOT / "finetune" / "_raw"

# Klassen-Mapping je Datensatz (Roboflow-Name → unsere class_id)
DATASETS = [
    {
        "workspace": "cocojumbo",
        "project":   "floorball-ku1s7",
        "version":   7,
        "format":    "yolov11",
        "map": {"player": 0, "Player": 0, "Balls": 2, "ball": 2},
    },
    {
        "workspace": "florbaltestset",
        "project":   "ball-model-cipht",
        "version":   3,
        "format":    "yolov11",
        "map": {"0": 2, "ball": 2, "Ball": 2},
    },
    {
        "workspace": "alsons-workspace",
        "project":   "floorball-tracker",
        "version":   2,
        "format":    "yolov11",
        "map": {"Player": 0, "player": 0, "Goalkeeper": 1, "goalkeeper": 1,
                "ball": 2, "Ball": 2},
        # Referee und objects werden nicht übernommen
    },
]


def _remap_labels(src_lbl: Path, dst_lbl: Path, id_map: dict[int, int]) -> int:
    """Remappt Klassen-IDs in einer YOLO-Label-Datei."""
    lines_out = []
    for line in src_lbl.read_text().strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        # Segmentation-Format hat mehr als 5 Werte → nur erste 5 (bbox) nehmen
        raw_cid = int(parts[0])
        new_cid = id_map.get(raw_cid)
        if new_cid is None:
            continue
        if len(parts) > 5:
            # Segmentation → bbox konvertieren (nur class + 4 bbox-Werte)
            # Punkte-Polygon überspringen, nur ersten 5 Werte nehmen falls vorhanden
            # Roboflow liefert bei yolov11-Format schon bbox
            bbox_parts = parts[1:5]
        else:
            bbox_parts = parts[1:]
        lines_out.append(f"{new_cid} {' '.join(bbox_parts)}")
    dst_lbl.write_text("\n".join(lines_out))
    return len(lines_out)


def download_all(api_key: str) -> None:
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)

    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_imgs = {"train": 0, "val": 0}

    for ds in DATASETS:
        tag = f"{ds['workspace']}/{ds['project']} v{ds['version']}"
        raw = RAW_DIR / ds["project"]
        print(f"\n{'='*55}")
        print(f"Lade: {tag}")

        proj = rf.workspace(ds["workspace"]).project(ds["project"])
        dataset = proj.version(ds["version"]).download(ds["format"],
                                                        location=str(raw))
        raw_dir = Path(dataset.location)

        # Klassen aus data.yaml lesen
        raw_yaml = raw_dir / "data.yaml"
        raw_names: list[str] = []
        if raw_yaml.exists():
            with open(raw_yaml) as f:
                cfg = yaml.safe_load(f)
            raw_names = cfg.get("names", [])
            print(f"  Roboflow-Klassen: {raw_names}")

        # Name → ID-Map aufbauen
        name_to_new: dict[int, int] = {}
        for raw_id, raw_name in enumerate(raw_names):
            new_id = ds["map"].get(raw_name)
            if new_id is not None:
                name_to_new[raw_id] = new_id
        # Fallback: direkte ID-Zuweisungen aus map (für ball-model "0")
        for k, v in ds["map"].items():
            try:
                name_to_new.setdefault(int(k), v)
            except ValueError:
                pass
        print(f"  ID-Mapping: {name_to_new}")

        # Bilder + Labels kopieren
        for rf_split, our_split in [("train", "train"), ("valid", "val"),
                                     ("val", "val"), ("test", "val")]:
            src_img = raw_dir / rf_split / "images"
            src_lbl = raw_dir / rf_split / "labels"
            if not src_img.exists():
                continue

            dst_img = DATASET_DIR / "images" / our_split
            dst_lbl = DATASET_DIR / "labels" / our_split

            n_img = 0
            for img_path in src_img.iterdir():
                dst = dst_img / f"{ds['project']}_{img_path.name}"
                shutil.copy2(img_path, dst)
                n_img += 1

                lbl_src = src_lbl / (img_path.stem + ".txt")
                lbl_dst = dst_lbl / f"{ds['project']}_{img_path.stem}.txt"
                if lbl_src.exists():
                    _remap_labels(lbl_src, lbl_dst, name_to_new)
                else:
                    lbl_dst.write_text("")

            total_imgs[our_split] += n_img
            print(f"  {our_split}: +{n_img} Bilder")

    print(f"\n{'='*55}")
    print(f"GESAMT: train={total_imgs['train']}  val={total_imgs['val']} Bilder")
    print(f"Dataset: {DATASET_DIR}")


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else API_KEY
    if not key:
        print("API-Key fehlt. Aufruf: python finetune/download_and_merge.py KEY")
        sys.exit(1)
    download_all(key)
