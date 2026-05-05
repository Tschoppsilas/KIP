"""Phase 6: Taktikboard-Renderer (anzeigeunabhängig, testbar).

Rendert alle taktischen Elemente (Spieler, Torwart, Ball, Laufwege, Pfeile)
auf eine PIL-Image-Kopie des Taktikboards.  Kein Tkinter/Display erforderlich –
die Klasse lässt sich direkt testen und auch für den PNG-Export (Phase 7) nutzen.

Koordinatensystem:
  - Feldkoordinaten (board coords) kommen aus der Homographie in Metern, z.B. (0–40, 0–20).
  - board_to_canvas() mappt diese auf Pixel-Koordinaten im Taktikboard-Bild.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Standardpfad zum Taktikboard-Hintergrundbild
# ---------------------------------------------------------------------------
_DEFAULT_BOARD = Path(__file__).parent.parent.parent / "Taktikboard" / "Taktikboard.png"

# ---------------------------------------------------------------------------
# Farben (R, G, B, A)
# ---------------------------------------------------------------------------
COLOR_TEAM_A     = (220,  40,  40, 230)   # Rot
COLOR_TEAM_B     = ( 30,  90, 220, 230)   # Blau
COLOR_GOALKEEPER = (180,  60, 220, 230)   # Lila
COLOR_BALL       = (240, 200,   0, 255)   # Gelb
COLOR_UNKNOWN    = (140, 140, 140, 200)
COLOR_PASS       = ( 20, 180,  20, 220)   # Grün
COLOR_SHOT       = (220,  60,  20, 220)   # Orange-Rot
COLOR_RUN        = ( 60, 160, 220, 200)   # Hellblau

TEAM_COLORS = {0: COLOR_TEAM_A, 1: COLOR_TEAM_B, -1: COLOR_UNKNOWN}

# ---------------------------------------------------------------------------
# Datenklassen für taktische Elemente
# ---------------------------------------------------------------------------

@dataclass
class PlayerSymbol:
    """Ein Spieler-Symbol auf dem Taktikboard.

    Attributes:
        track_id:   ByteTrack-ID.
        team:       0 = Team A, 1 = Team B, -1 = unbekannt.
        class_name: 'player', 'goalkeeper', 'ball'.
        board_x:    X-Position in Feldkoordinaten (Meter).
        board_y:    Y-Position in Feldkoordinaten (Meter).
        label:      Optionaler Beschriftungstext (Standard: track_id).
    """
    track_id: int
    team: int
    class_name: str
    board_x: float
    board_y: float
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = str(self.track_id)


@dataclass
class Arrow:
    """Ein taktischer Pfeil (Pass, Schuss, Laufweg).

    Attributes:
        x0, y0: Startpunkt in Feldkoordinaten.
        x1, y1: Endpunkt in Feldkoordinaten.
        kind:   'pass', 'shot', 'run'.
    """
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str = "pass"   # 'pass' | 'shot' | 'run'


@dataclass
class BoardState:
    """Vollständiger Zustand des Taktikboards zu einem Zeitpunkt.

    Attributes:
        players: Liste aller Spieler-Symbole.
        arrows:  Liste aller taktischen Pfeile.
        frame_index: Optionaler Frame-Index für die Beschriftung.
    """
    players: list[PlayerSymbol] = field(default_factory=list)
    arrows:  list[Arrow]        = field(default_factory=list)
    frame_index: int = -1


# ---------------------------------------------------------------------------
# BoardRenderer
# ---------------------------------------------------------------------------

class BoardRenderer:
    """Rendert ein BoardState auf das Taktikboard-Hintergrundbild.

    Args:
        board_image_path: Pfad zum Taktikboard-PNG.
        field_width_m:    Feldbreite in Metern (X-Achse, Standard 40 m).
        field_height_m:   Feldhöhe in Metern (Y-Achse, Standard 20 m).
        padding_frac:     Relativer Rand auf jeder Seite (0–0.5).
    """

    def __init__(
        self,
        board_image_path: str | Path = _DEFAULT_BOARD,
        field_width_m: float = 40.0,
        field_height_m: float = 20.0,
        padding_frac: float = 0.07,
    ) -> None:
        self._bg = Image.open(board_image_path).convert("RGBA")
        self.canvas_w, self.canvas_h = self._bg.size
        self.field_w = field_width_m
        self.field_h = field_height_m

        # Zeichenfläche innerhalb des Bildes (mit Rand)
        pad_x = int(self.canvas_w * padding_frac)
        pad_y = int(self.canvas_h * padding_frac)
        self._draw_x0 = pad_x
        self._draw_y0 = pad_y
        self._draw_x1 = self.canvas_w - pad_x
        self._draw_y1 = self.canvas_h - pad_y
        self._draw_w  = self._draw_x1 - self._draw_x0
        self._draw_h  = self._draw_y1 - self._draw_y0

    # ------------------------------------------------------------------
    # Koordinatentransformation
    # ------------------------------------------------------------------

    def board_to_canvas(self, bx: float, by: float) -> tuple[int, int]:
        """Feldkoordinaten (Meter) → Canvas-Pixel."""
        cx = self._draw_x0 + int(bx / self.field_w * self._draw_w)
        cy = self._draw_y0 + int(by / self.field_h * self._draw_h)
        return cx, cy

    def canvas_to_board(self, cx: int, cy: int) -> tuple[float, float]:
        """Canvas-Pixel → Feldkoordinaten (Meter)."""
        bx = (cx - self._draw_x0) / self._draw_w * self.field_w
        by = (cy - self._draw_y0) / self._draw_h * self.field_h
        return bx, by

    # ------------------------------------------------------------------
    # Render-Methode
    # ------------------------------------------------------------------

    def render(self, state: BoardState) -> Image.Image:
        """Zeichnet alle Elemente aus ``state`` auf eine Kopie des Hintergrundbildes.

        Args:
            state: Aktueller Board-Zustand.

        Returns:
            Neues PIL-Image (RGBA) mit allen Overlays.
        """
        img = self._bg.copy()
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Laufwege (als Linien hinter den Symbolen)
        for arrow in state.arrows:
            self._draw_arrow(draw, arrow)

        # Spieler-Symbole
        for player in state.players:
            self._draw_player(draw, player)

        # Frame-Info
        if state.frame_index >= 0:
            draw.text((10, 10), f"Frame {state.frame_index}",
                      fill=(40, 40, 40, 200))

        img.alpha_composite(overlay)
        return img

    def render_rgb(self, state: BoardState) -> Image.Image:
        """Wie ``render()``, gibt aber ein RGB-Bild zurück (für Tkinter / VideoWriter)."""
        return self.render(state).convert("RGB")

    # ------------------------------------------------------------------
    # Interne Zeichenmethoden
    # ------------------------------------------------------------------

    def _draw_player(self, draw: ImageDraw.ImageDraw, p: PlayerSymbol) -> None:
        cx, cy = self.board_to_canvas(p.board_x, p.board_y)

        if p.class_name == "ball":
            r = 9
            color = COLOR_BALL
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=(80, 60, 0, 255), width=2)
            return

        color = TEAM_COLORS.get(p.team, COLOR_UNKNOWN)

        # Negatives track_id = DET-only (ungetrackt) → kleinerer, transparenterer Kreis
        is_tracked = p.track_id >= 0

        if p.class_name == "goalkeeper":
            r = 16 if is_tracked else 11
            draw.rectangle((cx - r, cy - r, cx + r, cy + r),
                            fill=COLOR_GOALKEEPER, outline=(80, 0, 120, 255), width=2)
        else:
            r = 14 if is_tracked else 9
            # DET-only: niedrigere Deckkraft (Alpha 130 statt 230)
            if is_tracked:
                fill_color = color
                outline_color = (0, 0, 0, 200)
                outline_w = 2
            else:
                fill_color = (*color[:3], 140)  # semi-transparent
                outline_color = (0, 0, 0, 100)
                outline_w = 1
            draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                         fill=fill_color, outline=outline_color, width=outline_w)

        # Label nur bei getrackte Spieler (nicht bei DET-only)
        if is_tracked and p.label:
            label = p.label[:3]
            draw.text((cx, cy), label, fill=(255, 255, 255, 240), anchor="mm")

    def _draw_arrow(self, draw: ImageDraw.ImageDraw, arrow: Arrow) -> None:
        x0, y0 = self.board_to_canvas(arrow.x0, arrow.y0)
        x1, y1 = self.board_to_canvas(arrow.x1, arrow.y1)

        color_map = {"pass": COLOR_PASS, "shot": COLOR_SHOT, "run": COLOR_RUN}
        color = color_map.get(arrow.kind, COLOR_PASS)
        dash = arrow.kind == "run"

        if dash:
            self._draw_dashed_line(draw, x0, y0, x1, y1, color, width=3)
        else:
            draw.line((x0, y0, x1, y1), fill=color, width=3)

        # Pfeilspitze
        self._draw_arrowhead(draw, x0, y0, x1, y1, color, size=14)

    @staticmethod
    def _draw_arrowhead(
        draw: ImageDraw.ImageDraw,
        x0: int, y0: int, x1: int, y1: int,
        color: tuple, size: int = 12,
    ) -> None:
        angle = math.atan2(y1 - y0, x1 - x0)
        spread = math.radians(28)
        left  = (x1 - size * math.cos(angle - spread),
                 y1 - size * math.sin(angle - spread))
        right = (x1 - size * math.cos(angle + spread),
                 y1 - size * math.sin(angle + spread))
        draw.polygon([left, (x1, y1), right], fill=color)

    @staticmethod
    def _draw_dashed_line(
        draw: ImageDraw.ImageDraw,
        x0: int, y0: int, x1: int, y1: int,
        color: tuple, width: int = 2, dash_len: int = 12, gap_len: int = 8,
    ) -> None:
        total = math.hypot(x1 - x0, y1 - y0)
        if total < 1:
            return
        dx, dy = (x1 - x0) / total, (y1 - y0) / total
        pos = 0.0
        drawing = True
        while pos < total:
            seg = dash_len if drawing else gap_len
            end = min(pos + seg, total)
            if drawing:
                sx0, sy0 = int(x0 + dx * pos), int(y0 + dy * pos)
                sx1, sy1 = int(x0 + dx * end), int(y0 + dy * end)
                draw.line((sx0, sy0, sx1, sy1), fill=color, width=width)
            pos = end
            drawing = not drawing
