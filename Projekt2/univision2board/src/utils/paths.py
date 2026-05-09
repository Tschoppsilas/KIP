"""
Zentrale Pfad-Definitionen für UniVision2Board.

Projektroot ist das Verzeichnis, das `univision2board/` enthält.
Alle Ausgabepfade werden relativ dazu aufgelöst und bei Bedarf erstellt.
"""

import os

# Projektroot = Elternverzeichnis von univision2board/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # src/utils/
_SRC_DIR = os.path.dirname(_THIS_DIR)                           # src/
_PACKAGE_DIR = os.path.dirname(_SRC_DIR)                        # univision2board/
PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)                    # KIP_1_neuer versuch/

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
OUTPUT_EXPORTS = os.path.join(OUTPUT_ROOT, "exports")           # PNG / PDF
OUTPUT_VIDEO = os.path.join(OUTPUT_ROOT, "video")               # MP4


def ensure_output_dirs() -> None:
    """Erstellt alle Ausgabe-Verzeichnisse, falls sie nicht existieren."""
    for d in (OUTPUT_EXPORTS, OUTPUT_VIDEO):
        os.makedirs(d, exist_ok=True)


def export_path(filename: str) -> str:
    """Absoluter Pfad für eine PNG/PDF-Datei in output/exports/."""
    os.makedirs(OUTPUT_EXPORTS, exist_ok=True)
    return os.path.join(OUTPUT_EXPORTS, filename)


def video_path(filename: str) -> str:
    """Absoluter Pfad für eine MP4-Datei in output/video/."""
    os.makedirs(OUTPUT_VIDEO, exist_ok=True)
    return os.path.join(OUTPUT_VIDEO, filename)
