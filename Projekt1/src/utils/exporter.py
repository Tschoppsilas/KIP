"""Phase 7: Datenexport – PNG, PDF, Video-Overlay, temporäre Vorschau.

Alle Funktionen sind unabhängig von Tkinter und lassen sich direkt testen.

Export-Typen:
  - PNG  : verlustfreies Rasterbild (Standard)
  - PDF  : eingebettetes Rasterbild in PDF-Seite (via Pillow)
  - Video: annotiertes MP4 mit allen sichtbaren Overlays (via OpenCV)
  - Temp : temporäre Vorschau ohne persistente Datei (Should)

Could: ``ExportPreset`` fasst Qualitäts- und Formateinstellungen zusammen.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Could: Export-Presets
# ---------------------------------------------------------------------------

@dataclass
class ExportPreset:
    """Fasst Qualitäts- und Formateinstellungen für einen Export zusammen.

    Attributes:
        format:       Ausgabeformat: 'png' | 'pdf' | 'mp4'.
        png_compress: PNG-Kompression 0 (keine) … 9 (maximal). Standard 3.
        pdf_dpi:      DPI-Angabe in den PDF-Metadaten. Standard 150.
        video_fps:    Frames pro Sekunde für den Video-Export. Standard 30.
        video_codec:  FourCC-Codec-String für cv2.VideoWriter. Standard 'mp4v'.
    """
    format: str = "png"
    png_compress: int = 3
    pdf_dpi: int = 150
    video_fps: float = 30.0
    video_codec: str = "mp4v"


# Vordefinierte Presets
PRESET_DEFAULT  = ExportPreset(format="png")
PRESET_HIGHRES  = ExportPreset(format="png", png_compress=0)
PRESET_PDF      = ExportPreset(format="pdf", pdf_dpi=150)
PRESET_VIDEO_HD = ExportPreset(format="mp4", video_fps=30.0, video_codec="mp4v")


# ---------------------------------------------------------------------------
# Bildexport (PNG / PDF)
# ---------------------------------------------------------------------------

def export_png(
    image: Image.Image,
    path: str | Path,
    compress: int = 3,
) -> Path:
    """Exportiert ein PIL-Image als PNG-Datei.

    Args:
        image:    Quell-Image (RGB oder RGBA).
        path:     Zielpfad (Endung wird auf .png gesetzt, falls abweichend).
        compress: PNG-Kompression 0–9.

    Returns:
        Absoluter Pfad der erzeugten Datei.
    """
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(str(path), format="PNG", compress_level=compress)
    return path.resolve()


def export_pdf(
    image: Image.Image,
    path: str | Path,
    dpi: int = 150,
) -> Path:
    """Exportiert ein PIL-Image als einseitige PDF-Datei.

    Args:
        image: Quell-Image.
        path:  Zielpfad (Endung wird auf .pdf gesetzt).
        dpi:   DPI-Angabe in den PDF-Metadaten.

    Returns:
        Absoluter Pfad der erzeugten Datei.
    """
    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(str(path), format="PDF", resolution=dpi)
    return path.resolve()


def export_image(
    image: Image.Image,
    path: str | Path,
    preset: ExportPreset | None = None,
) -> Path:
    """Exportiert ein Image gemäss dem angegebenen Preset.

    Wählt automatisch ``export_png`` oder ``export_pdf`` anhand von
    ``preset.format`` (oder der Dateiendung von ``path``).

    Args:
        image:  Quell-Image.
        path:   Zielpfad.
        preset: Optionales ExportPreset. Standard: PRESET_DEFAULT (PNG).

    Returns:
        Absoluter Pfad der erzeugten Datei.
    """
    p = preset or PRESET_DEFAULT
    suffix = Path(path).suffix.lower()
    if p.format == "pdf" or suffix == ".pdf":
        return export_pdf(image, path, dpi=p.pdf_dpi)
    return export_png(image, path, compress=p.png_compress)


# ---------------------------------------------------------------------------
# Videoexport (cv2.VideoWriter)
# ---------------------------------------------------------------------------

def export_video(
    frames: Sequence[Image.Image | np.ndarray],
    path: str | Path,
    fps: float = 30.0,
    codec: str = "mp4v",
) -> Path:
    """Exportiert eine Folge von Frames als MP4-Video.

    Args:
        frames: Geordnete Liste von PIL-Images oder NumPy-BGR-Arrays.
        path:   Zielpfad (Endung wird auf .mp4 gesetzt).
        fps:    Frames pro Sekunde.
        codec:  FourCC-String, z.B. 'mp4v' oder 'avc1'.

    Returns:
        Absoluter Pfad der erzeugten Datei.

    Raises:
        ValueError:  Falls ``frames`` leer ist.
        RuntimeError: Falls der VideoWriter nicht geöffnet werden kann.
    """
    if not frames:
        raise ValueError("frames darf nicht leer sein.")

    path = Path(path).with_suffix(".mp4")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Grösse aus dem ersten Frame bestimmen
    first = frames[0]
    if isinstance(first, Image.Image):
        w, h = first.size
    else:
        h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter konnte nicht geöffnet werden: {path}")

    for frame in frames:
        if isinstance(frame, Image.Image):
            bgr = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            bgr = frame
        writer.write(bgr)

    writer.release()
    return path.resolve()


def export_video_from_states(
    states,           # list[BoardState]
    renderer,         # BoardRenderer
    path: str | Path,
    preset: ExportPreset | None = None,
) -> Path:
    """Rendert alle BoardStates und exportiert sie als Video.

    Args:
        states:   Liste von BoardState-Objekten (Phase 6).
        renderer: BoardRenderer-Instanz.
        path:     Zielpfad.
        preset:   Optionales ExportPreset.

    Returns:
        Absoluter Pfad des erzeugten Videos.
    """
    p = preset or PRESET_VIDEO_HD
    frames = [renderer.render_rgb(s) for s in states]
    return export_video(frames, path, fps=p.video_fps, codec=p.video_codec)


# ---------------------------------------------------------------------------
# Should: Temporäre Vorschau ohne persistente Speicherung
# ---------------------------------------------------------------------------

def preview_temp(
    image: Image.Image,
    open_viewer: bool = True,
) -> None:
    """Zeigt ein Image in der temporären Systemvorschau an.

    Schreibt das Bild in eine temporäre PNG-Datei, öffnet sie mit dem
    Standard-Bildbetrachter (``xdg-open``) und löscht die Datei anschliessend
    nicht sofort – das Betriebssystem räumt das Temp-Verzeichnis selbst auf.
    Erzeugt dadurch **keine persistente Ausgabedatei** im Projektverzeichnis.

    Args:
        image:       Anzuzeigendes Image.
        open_viewer: Falls False, wird kein Betrachter geöffnet (für Tests).

    Returns:
        None
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False,
                                    prefix="univision_preview_") as tmp:
        tmp_path = tmp.name

    image.convert("RGB").save(tmp_path, format="PNG")

    if open_viewer:
        subprocess.Popen(["xdg-open", tmp_path],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
