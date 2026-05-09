"""
TacticBoardWidget – Haupt-Canvas des Taktikboards.

Zeigt:
  - Hintergrundbild (Taktikboard.png)
  - Spielerpositionen als farbige Kreise (Team A=rot, Team B=blau)
  - Laufwege der Spieler (Linien in Teamfarbe)
  - Regelbasierte Passvorschläge (gestrichelte Pfeile)
  - Manuell gezeichnete Elemente (Pass, Schuss, Laufweg)

Koordinatensystem: Board-Image-Pixel (0..BOARD_W, 0..BOARD_H).
Beim Rendern wird in Widget-Pixel skaliert.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import (
    QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF,
)
from PyQt5.QtWidgets import QSizePolicy, QWidget

from src.gui.pass_suggester import PassSuggestion
from src.utils import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Konstanten
# ------------------------------------------------------------------
TEAM_COLORS = {
    0: QColor(210, 50, 50),   # Team A – Rot
    1: QColor(50, 100, 220),  # Team B – Blau
}
PLAYER_RADIUS = 10
TRAJ_MAX_POINTS = 50  # Letzten N Punkte im Laufweg zeichnen
ARROW_HEAD_SIZE = 10


# ------------------------------------------------------------------
# Datenklassen für manuell gezeichnete Elemente
# ------------------------------------------------------------------
class DrawMode(Enum):
    SELECT = auto()
    DRAW_PASS = auto()
    DRAW_SHOT = auto()
    DRAW_PATH = auto()


@dataclass
class TacticElement:
    """Ein vom Trainer manuell gezeichnetes taktisches Element."""
    kind: str  # "pass" | "shot" | "path"
    points: List[Tuple[float, float]] = field(default_factory=list)


# ------------------------------------------------------------------
# Widget
# ------------------------------------------------------------------
class TacticBoardWidget(QWidget):
    """
    Interaktives Taktikboard.

    Signals:
        player_clicked(track_id): ein Spieler wurde angeklickt.
    """

    player_clicked = pyqtSignal(int)

    def __init__(self, board_path: str, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)

        # Hintergrundbild laden
        img = cv2.imread(board_path)
        if img is None:
            raise FileNotFoundError(f"Taktikboard nicht gefunden: {board_path}")
        self._board_w = img.shape[1]
        self._board_h = img.shape[0]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888)
        self._board_pixmap = QPixmap.fromImage(qimg)

        # Zustand
        self._player_positions: Dict[int, Tuple[float, float]] = {}
        self._player_teams: Dict[int, int] = {}
        self._trajectories: Dict[int, List[Tuple[float, float]]] = {}
        self._pass_suggestions: List[PassSuggestion] = []
        self._drawn_elements: List[TacticElement] = []
        self._selected_ids: set = set()
        self._show_trajectories: bool = True
        self._show_pass_suggestions: bool = True

        # Zeichenmodus
        self._draw_mode: DrawMode = DrawMode.SELECT
        self._current_draw_points: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Öffentliche API – Zustand setzen
    # ------------------------------------------------------------------

    def update_frame(
        self,
        positions: Dict[int, Tuple[float, float]],
        teams: Dict[int, int],
        trajectories: Dict[int, List[Tuple[float, float]]],
        pass_suggestions: List[PassSuggestion],
    ) -> None:
        """Aktualisiert alle Spielerdaten für den aktuellen Frame."""
        self._player_positions = dict(positions)
        self._player_teams = dict(teams)
        self._trajectories = dict(trajectories)
        self._pass_suggestions = list(pass_suggestions)
        self.update()

    def set_draw_mode(self, mode: DrawMode) -> None:
        self._draw_mode = mode
        self._current_draw_points.clear()
        self.update()

    def set_show_trajectories(self, visible: bool) -> None:
        self._show_trajectories = visible
        self.update()

    def set_show_pass_suggestions(self, visible: bool) -> None:
        self._show_pass_suggestions = visible
        self.update()

    def clear_drawn_elements(self) -> None:
        self._drawn_elements.clear()
        self.update()

    def get_drawn_elements(self) -> List[TacticElement]:
        return list(self._drawn_elements)

    # ------------------------------------------------------------------
    # Koordinaten-Umrechnung (Board ↔ Widget)
    # ------------------------------------------------------------------

    def _scale(self) -> Tuple[float, float, float, float]:
        """Gibt (scale_x, scale_y, offset_x, offset_y) zurück."""
        pix = self._board_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        ox = (self.width() - pix.width()) / 2
        oy = (self.height() - pix.height()) / 2
        sx = pix.width() / self._board_w
        sy = pix.height() / self._board_h
        return sx, sy, ox, oy

    def _board_to_widget(self, bx: float, by: float) -> Tuple[float, float]:
        sx, sy, ox, oy = self._scale()
        return bx * sx + ox, by * sy + oy

    def _widget_to_board(self, wx: float, wy: float) -> Tuple[float, float]:
        sx, sy, ox, oy = self._scale()
        return (wx - ox) / sx, (wy - oy) / sy

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Hintergrund
        pix = self._board_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        sx, sy, ox, oy = self._scale()
        painter.drawPixmap(int(ox), int(oy), pix)

        # Laufwege
        if self._show_trajectories:
            self._draw_trajectories(painter)

        # Passvorschläge
        if self._show_pass_suggestions:
            self._draw_pass_suggestions(painter)

        # Manuell gezeichnete Elemente
        self._draw_elements(painter)

        # Spieler
        self._draw_players(painter)

        # Aktuell in Arbeit (Zeichenmodus)
        self._draw_in_progress(painter)

        painter.end()

    def _draw_players(self, painter: QPainter) -> None:
        for tid, (bx, by) in self._player_positions.items():
            wx, wy = self._board_to_widget(bx, by)
            team = self._player_teams.get(tid, 0)
            color = TEAM_COLORS.get(team, QColor(150, 150, 150))
            selected = tid in self._selected_ids

            # Kreis
            if selected:
                painter.setPen(QPen(Qt.white, 3))
            else:
                painter.setPen(QPen(color.darker(150), 1))
            painter.setBrush(color)
            painter.drawEllipse(
                QPointF(wx, wy), PLAYER_RADIUS, PLAYER_RADIUS
            )

            # Track-ID als Label
            painter.setPen(QPen(Qt.white, 1))
            font = QFont("Arial", 6, QFont.Bold)
            painter.setFont(font)
            painter.drawText(
                int(wx - PLAYER_RADIUS + 1), int(wy + 4),
                str(tid)
            )

    def _draw_trajectories(self, painter: QPainter) -> None:
        for tid, points in self._trajectories.items():
            if len(points) < 2:
                continue
            team = self._player_teams.get(tid, 0)
            color = TEAM_COLORS.get(team, QColor(150, 150, 150))
            pen = QPen(color, 1, Qt.SolidLine)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            pts = points[-TRAJ_MAX_POINTS:]
            for i in range(1, len(pts)):
                wx0, wy0 = self._board_to_widget(*pts[i - 1])
                wx1, wy1 = self._board_to_widget(*pts[i])
                painter.drawLine(QPointF(wx0, wy0), QPointF(wx1, wy1))

    def _draw_pass_suggestions(self, painter: QPainter) -> None:
        pen = QPen(QColor(0, 220, 100), 2, Qt.DashLine)
        painter.setPen(pen)
        for sug in self._pass_suggestions:
            wx0, wy0 = self._board_to_widget(*sug.from_pos)
            wx1, wy1 = self._board_to_widget(*sug.to_pos)
            painter.drawLine(QPointF(wx0, wy0), QPointF(wx1, wy1))
            self._draw_arrowhead(painter, wx0, wy0, wx1, wy1, QColor(0, 220, 100))

    def _draw_elements(self, painter: QPainter) -> None:
        for el in self._drawn_elements:
            if el.kind == "pass":
                self._draw_arrow_element(painter, el, QColor(255, 220, 0), Qt.SolidLine)
            elif el.kind == "shot":
                self._draw_arrow_element(painter, el, QColor(255, 80, 0), Qt.SolidLine)
            elif el.kind == "path":
                self._draw_path_element(painter, el, QColor(180, 100, 255))

    def _draw_arrow_element(self, painter, el: TacticElement, color: QColor, style) -> None:
        if len(el.points) < 2:
            return
        wx0, wy0 = self._board_to_widget(*el.points[0])
        wx1, wy1 = self._board_to_widget(*el.points[1])
        painter.setPen(QPen(color, 2, style))
        painter.drawLine(QPointF(wx0, wy0), QPointF(wx1, wy1))
        self._draw_arrowhead(painter, wx0, wy0, wx1, wy1, color)

    def _draw_path_element(self, painter, el: TacticElement, color: QColor) -> None:
        if len(el.points) < 2:
            return
        painter.setPen(QPen(color, 2, Qt.DotLine))
        for i in range(1, len(el.points)):
            wx0, wy0 = self._board_to_widget(*el.points[i - 1])
            wx1, wy1 = self._board_to_widget(*el.points[i])
            painter.drawLine(QPointF(wx0, wy0), QPointF(wx1, wy1))

    def _draw_in_progress(self, painter: QPainter) -> None:
        if len(self._current_draw_points) < 1:
            return
        color = {
            DrawMode.DRAW_PASS: QColor(255, 220, 0),
            DrawMode.DRAW_SHOT: QColor(255, 80, 0),
            DrawMode.DRAW_PATH: QColor(180, 100, 255),
        }.get(self._draw_mode, QColor(200, 200, 200))

        painter.setPen(QPen(color, 2, Qt.DashLine))
        pts_w = [self._board_to_widget(*p) for p in self._current_draw_points]
        for i in range(1, len(pts_w)):
            painter.drawLine(QPointF(*pts_w[i - 1]), QPointF(*pts_w[i]))

    @staticmethod
    def _draw_arrowhead(
        painter: QPainter,
        x0: float, y0: float, x1: float, y1: float,
        color: QColor,
    ) -> None:
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        dx, dy = dx / length, dy / length
        size = ARROW_HEAD_SIZE
        lx = x1 - size * (dx + 0.4 * dy)
        ly = y1 - size * (dy - 0.4 * dx)
        rx = x1 - size * (dx - 0.4 * dy)
        ry = y1 - size * (dy + 0.4 * dx)
        painter.setPen(QPen(color, 1))
        painter.setBrush(color)
        painter.drawPolygon(
            QPolygonF([QPointF(x1, y1), QPointF(lx, ly), QPointF(rx, ry)])
        )

    # ------------------------------------------------------------------
    # Mauseingabe
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        bx, by = self._widget_to_board(event.x(), event.y())

        if self._draw_mode == DrawMode.SELECT:
            self._handle_select(bx, by)
        elif self._draw_mode in (DrawMode.DRAW_PASS, DrawMode.DRAW_SHOT):
            self._handle_arrow_click(bx, by)
        elif self._draw_mode == DrawMode.DRAW_PATH:
            self._handle_path_click(bx, by)

    def mouseDoubleClickEvent(self, event):
        """Doppelklick beendet den Laufweg-Zeichenmodus."""
        if self._draw_mode == DrawMode.DRAW_PATH and self._current_draw_points:
            el = TacticElement(kind="path", points=list(self._current_draw_points))
            self._drawn_elements.append(el)
            self._current_draw_points.clear()
            self.update()

    def _handle_select(self, bx: float, by: float) -> None:
        hit = self._hit_player(bx, by)
        if hit is not None:
            if hit in self._selected_ids:
                self._selected_ids.discard(hit)
            else:
                self._selected_ids.add(hit)
            self.player_clicked.emit(hit)
        else:
            self._selected_ids.clear()
        self.update()

    def _handle_arrow_click(self, bx: float, by: float) -> None:
        self._current_draw_points.append((bx, by))
        if len(self._current_draw_points) == 2:
            kind = "pass" if self._draw_mode == DrawMode.DRAW_PASS else "shot"
            el = TacticElement(kind=kind, points=list(self._current_draw_points))
            self._drawn_elements.append(el)
            self._current_draw_points.clear()
        self.update()

    def _handle_path_click(self, bx: float, by: float) -> None:
        self._current_draw_points.append((bx, by))
        self.update()

    def _hit_player(self, bx: float, by: float) -> Optional[int]:
        """Gibt die Track-ID des angeklickten Spielers zurück."""
        sx, sy, _, _ = self._scale()
        hit_radius = PLAYER_RADIUS / min(sx, sy) * 1.5
        for tid, (px, py) in self._player_positions.items():
            if math.hypot(bx - px, by - py) <= hit_radius:
                return tid
        return None
