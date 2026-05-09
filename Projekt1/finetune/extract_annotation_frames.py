"""Extrahiert Frames aus allen Videos für manuelles Labeling auf Roboflow.

Klassen (neu):
  0 = player
  1 = goalkeeper
  2 = ball
  3 = referee   ← NEU

Aufruf:
    python finetune/extract_annotation_frames.py --n 200 --out finetune/annotation_frames_v2
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).parent.parent


def extract_frames(
    video_paths: list[Path],
    n_total: int,
    out_dir: Path,
) -> int:
    """Extrahiert n_total Frames gleichmässig über alle Videos.

    Pro Video werden gleich viele Frames aus verschiedenen Zeitpunkten
    (gleichmässig verteilt über die gesamte Länge) entnommen.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n_per_video = max(1, n_total // len(video_paths))
    remainder   = n_total - n_per_video * len(video_paths)

    saved = 0
    for vi, vp in enumerate(video_paths):
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"[WARN] Nicht lesbar: {vp.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_extract = n_per_video + (1 if vi < remainder else 0)
        # gleichmässige Frame-Indizes über das gesamte Video
        step = max(1, total_frames // n_extract)
        indices = [i * step for i in range(n_extract)]

        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            fname = f"{vp.stem}_{fi:06d}.jpg"
            cv2.imwrite(str(out_dir / fname), frame)
            saved += 1

        cap.release()
        print(f"  {vp.name:30s}  → {n_extract} Frames")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=200,
                        help="Anzahl Frames total (default 200)")
    parser.add_argument("--out", default="finetune/annotation_frames_v2",
                        help="Ausgabeordner")
    parser.add_argument("--zip", action="store_true",
                        help="ZIP-Archiv erstellen")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out
    if out_dir.exists():
        shutil.rmtree(out_dir)

    videos = sorted((PROJECT_ROOT / "Videos" / "Trainingsdaten").glob("*.mp4"))
    if not videos:
        print("Keine Videos gefunden unter Videos/Trainingsdaten/")
        return

    print(f"\n{len(videos)} Videos gefunden – extrahiere {args.n} Frames …\n")
    saved = extract_frames(videos, args.n, out_dir)

    print(f"\n✓ {saved} Frames gespeichert → {out_dir}")

    if args.zip:
        zip_path = PROJECT_ROOT / "finetune" / "annotation_frames_v2.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for img in sorted(out_dir.glob("*.jpg")):
                zf.write(img, img.name)
        print(f"✓ ZIP erstellt: {zip_path}")

    print("""
Nächste Schritte auf Roboflow:
──────────────────────────────
1. Projekt öffnen (oder neues Projekt anlegen)
2. Bilder hochladen aus: """ + str(out_dir) + """
3. Klassen definieren:
     player, goalkeeper, ball, referee
4. Annotieren (Auto-Label + manuelle Korrekturen)
   → Schiedsrichter als 'referee' labeln
   → Personen ausserhalb des Feldes NICHT labeln
5. Export: YOLOv8 Format → ZIP herunterladen
6. ZIP ablegen unter: finetune/roboflow_export_v2.zip
""")


if __name__ == "__main__":
    main()
