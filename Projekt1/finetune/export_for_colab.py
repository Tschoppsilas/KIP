"""Exportiert den Datensatz als ZIP-Datei für Google Colab Training.

Aufruf:
    python finetune/export_for_colab.py

Erstellt:
    colab_dataset.zip  (im Projektroot)
    → enthält: finetune/dataset/ + finetune/data.yaml
"""

from __future__ import annotations
import zipfile
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DS_DIR    = ROOT / "finetune" / "dataset"
DATA_YAML = ROOT / "finetune" / "data.yaml"
OUT_ZIP   = ROOT / "colab_dataset.zip"

def main() -> None:
    print("Packe Datensatz für Google Colab...")

    # Alle Bilder und Labels zählen
    train_imgs = list((DS_DIR / "images" / "train").glob("*"))
    val_imgs   = list((DS_DIR / "images" / "val").glob("*"))
    train_lbls = list((DS_DIR / "labels" / "train").glob("*.txt"))
    val_lbls   = list((DS_DIR / "labels" / "val").glob("*.txt"))

    print(f"  Train: {len(train_imgs)} Bilder, {len(train_lbls)} Labels")
    print(f"  Val:   {len(val_imgs)} Bilder, {len(val_lbls)} Labels")

    files: list[Path] = train_imgs + val_imgs + train_lbls + val_lbls + [DATA_YAML]

    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arc = f.relative_to(ROOT / "finetune")
            zf.write(f, arc)
            print(f"  + {arc}", end="\r")

    size_mb = OUT_ZIP.stat().st_size / 1_048_576
    print(f"\nFertig: {OUT_ZIP}  ({size_mb:.1f} MB)")
    print()
    print("Nächster Schritt:")
    print("  1. Lade 'colab_dataset.zip' in dein Google Drive hoch")
    print("     (Ordner: Mein Drive / KIP_Training/)")
    print("  2. Öffne das Colab-Notebook: finetune/colab_training.ipynb")
    print("  3. Klicke 'Laufzeit > Alle ausführen'")

if __name__ == "__main__":
    main()
