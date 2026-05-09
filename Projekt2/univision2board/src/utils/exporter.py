"""
Exportfunktionen: Taktikboard als PNG/PDF und annotiertes Video via OpenCV.

Bildexport:
  - PNG: direkt via OpenCV (cv2.imwrite)
  - PDF: via reportlab (Bild eingebettet in A4/custom-Seite)

Videoexport:
  - Jeder Frame wird mit Spieler-Overlays (Kreise, IDs, Laufwege) versehen
    und via cv2.VideoWriter als MP4 gespeichert.
  - Split-View optional: Original-Frame links, Taktikboard rechts.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Team-Farben für Video-Overlays (BGR)
_TEAM_BGR = {
    0: (50, 50, 210),   # Team A – Rot
    1: (220, 100, 50),  # Team B – Blau
}
_TRAJ_MAX = 40          # Letzten N Laufwegpunkte zeichnen


# ---------------------------------------------------------------------------
# Bildexport
# ---------------------------------------------------------------------------

def export_png(image: np.ndarray, path: str) -> None:
    """Speichert ein BGR-Bild als PNG."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ok = cv2.imwrite(path, image)
    if not ok:
        raise IOError(f"PNG-Export fehlgeschlagen: {path}")
    logger.info(f"PNG gespeichert: {path}")


def export_pdf(image: np.ndarray, path: str, title: str = "Taktikboard") -> None:
    """
    Bettet ein BGR-Bild in eine PDF-Seite ein.
    Die Seitengrösse entspricht der Bildgrösse in Punkt (72 dpi).
    """
    from reportlab.pdfgen import canvas as rl_canvas
    import tempfile

    # 1 pt = 1 reportlab-Einheit (interne Basiseinheit)
    PT = 1.0

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Temporäres PNG als Zwischenstufe
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, image)
        h, w = image.shape[:2]
        c = rl_canvas.Canvas(path, pagesize=(w * PT, h * PT))
        c.setTitle(title)
        c.drawImage(tmp_path, 0, 0, width=w * PT, height=h * PT)
        c.save()
    finally:
        os.unlink(tmp_path)

    logger.info(f"PDF gespeichert: {path}")


# ---------------------------------------------------------------------------
# Board-Overlay-Renderer (für Videoexport)
# ---------------------------------------------------------------------------

def render_board_frame(
    board_img: np.ndarray,
    positions: Dict[int, Tuple[float, float]],
    teams: Dict[int, int],
    trajectories: Dict[int, List[Tuple[float, float]]],
) -> np.ndarray:
    """
    Zeichnet Spieler-Overlays auf eine Kopie des Taktikboard-Bilds.

    Returns:
        BGR-Bild mit Spielern und Laufwegen.
    """
    out = board_img.copy()

    # Laufwege zuerst (unter den Spieler-Kreisen)
    for tid, pts in trajectories.items():
        if len(pts) < 2:
            continue
        team = teams.get(tid, 0)
        color = _TEAM_BGR.get(team, (180, 180, 180))
        recent = pts[-_TRAJ_MAX:]
        for i in range(1, len(recent)):
            p0 = (int(recent[i - 1][0]), int(recent[i - 1][1]))
            p1 = (int(recent[i][0]), int(recent[i][1]))
            cv2.line(out, p0, p1, color, 1, cv2.LINE_AA)

    # Spieler-Kreise
    for tid, (bx, by) in positions.items():
        team = teams.get(tid, 0)
        color = _TEAM_BGR.get(team, (180, 180, 180))
        center = (int(bx), int(by))
        cv2.circle(out, center, 10, color, -1, cv2.LINE_AA)
        cv2.circle(out, center, 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            out, str(tid),
            (center[0] - 6, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return out


# ---------------------------------------------------------------------------
# Videoexport
# ---------------------------------------------------------------------------

class VideoExporter:
    """
    Schreibt annotierte Frames als MP4-Datei (H.264, mp4v-Fallback).

    Parameters:
        output_path:  Zielpfad der .mp4-Datei.
        fps:          Bildrate (sollte der des Quellvideos entsprechen).
        split_view:   True → linke Hälfte Original-Frame, rechte Hälfte Board.
                      False → nur Board-Overlay.
    """

    def __init__(self, output_path: str, fps: float = 30.0, split_view: bool = True):
        self.output_path = output_path
        self.fps = fps
        self.split_view = split_view
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count = 0
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def _init_writer(self, width: int, height: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        if not self._writer.isOpened():
            raise IOError(f"VideoWriter konnte nicht geöffnet werden: {self.output_path}")
        logger.info(f"VideoExporter geöffnet: {self.output_path} ({width}x{height} @ {self.fps} fps)")

    def write_frame(
        self,
        board_frame: np.ndarray,
        video_frame: Optional[np.ndarray] = None,
    ) -> None:
        """
        Schreibt einen annotierten Frame.

        Parameters:
            board_frame: Board-Overlay-Bild (BGR, bereits mit render_board_frame gerendert).
            video_frame: Original-Videoframe für Split-View (optional).
        """
        if self.split_view and video_frame is not None:
            bh, bw = board_frame.shape[:2]
            vh, vw = video_frame.shape[:2]
            # Beide auf gleiche Höhe skalieren
            if vh != bh:
                video_frame = cv2.resize(video_frame, (int(vw * bh / vh), bh))
            combined = np.hstack([video_frame, board_frame])
            out_frame = combined
        else:
            out_frame = board_frame

        if self._writer is None:
            h, w = out_frame.shape[:2]
            self._init_writer(w, h)

        self._writer.write(out_frame)
        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(f"VideoExporter geschlossen: {self._frame_count} Frames → {self.output_path}")

    def __enter__(self) -> "VideoExporter":
        return self

    def __exit__(self, *args) -> None:
        self.close()
