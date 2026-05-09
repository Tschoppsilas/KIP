"""
Team-Korrektur-Dialog: Trainer kann falsch zugeordnete Spieler umklicken.

Zeigt alle getrackten Spieler als Bildausschnitt (Crop) in einem Raster.
Jeder Spieler ist farblich seiner aktuellen Teamzuordnung entsprechend
markiert (rot = Team A, blau = Team B).  Klick auf einen Spieler wechselt
sein Team sofort; "Bestätigen" übernimmt alle Änderungen.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QFont
from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.tracking.team_assigner import TEAM_A, TEAM_B, TEAM_NAMES
from src.utils import get_logger

logger = get_logger(__name__)

# Team-Farben für Rahmen und Labels
TEAM_COLORS = {
    TEAM_A: QColor(210, 50, 50),   # Rot
    TEAM_B: QColor(50, 100, 220),  # Blau
}
TEAM_BG = {
    TEAM_A: "#8b0000",
    TEAM_B: "#00008b",
}

CROP_SIZE = 80   # px für die Spieler-Thumbnails
COLS = 6         # Thumbnails pro Reihe


class _PlayerTile(QWidget):
    """Einzelner Spieler-Kachel: Crop + Team-Label, klickbar."""

    def __init__(self, track_id: int, crop: np.ndarray, team: int, parent=None):
        super().__init__(parent)
        self.track_id = track_id
        self.team = team
        self._crop = crop
        self._build()
        self.setFixedSize(CROP_SIZE + 8, CROP_SIZE + 28)
        self.setCursor(Qt.PointingHandCursor)

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._img_lbl = QLabel()
        self._img_lbl.setFixedSize(CROP_SIZE, CROP_SIZE)
        self._img_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._img_lbl)

        self._team_lbl = QLabel()
        self._team_lbl.setAlignment(Qt.AlignCenter)
        self._team_lbl.setFixedHeight(18)
        font = QFont("Arial", 7, QFont.Bold)
        self._team_lbl.setFont(font)
        layout.addWidget(self._team_lbl)

        self._refresh()

    def _refresh(self):
        color = TEAM_COLORS[self.team]
        bg = TEAM_BG[self.team]

        # Crop → QPixmap mit farbigem Rahmen
        h, w = self._crop.shape[:2]
        crop_rgb = cv2.cvtColor(self._crop, cv2.COLOR_BGR2RGB)
        qimg = QImage(crop_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            CROP_SIZE, CROP_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        canvas = QPixmap(CROP_SIZE, CROP_SIZE)
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        x_off = (CROP_SIZE - pix.width()) // 2
        y_off = (CROP_SIZE - pix.height()) // 2
        painter.drawPixmap(x_off, y_off, pix)
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(1, 1, CROP_SIZE - 3, CROP_SIZE - 3)
        painter.end()
        self._img_lbl.setPixmap(canvas)

        self._team_lbl.setText(f"ID {self.track_id} · {TEAM_NAMES[self.team]}")
        self._team_lbl.setStyleSheet(
            f"color: white; background: {bg}; border-radius: 3px; padding: 1px 3px;"
        )

    def toggle_team(self):
        self.team = TEAM_B if self.team == TEAM_A else TEAM_A
        self._refresh()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_team()


class TeamCorrectionDialog(QDialog):
    """
    Dialog zur manuellen Korrektur der Team-Zuordnung.

    Parameters:
        player_data: Liste von (track_id, crop_bgr, initial_team).
        parent:      Eltern-Widget.
    """

    def __init__(
        self,
        player_data: List[Tuple[int, np.ndarray, int]],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Team-Korrektur – Spieler klicken zum Umschalten")
        self.setMinimumSize(600, 400)
        self._tiles: List[_PlayerTile] = []
        self._build_ui(player_data)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, player_data: List[Tuple[int, np.ndarray, int]]) -> None:
        root = QVBoxLayout(self)

        # Anleitung
        info = QLabel(
            "Klicke auf einen Spieler, um sein Team zu wechseln  "
            "(Rot = Team A · Blau = Team B)."
        )
        info.setStyleSheet(
            "font-size: 12px; padding: 6px; "
            "background: #1e1e2e; color: #cdd6f4; border-radius: 4px;"
        )
        info.setAlignment(Qt.AlignCenter)
        root.addWidget(info)

        # Scrollbares Raster
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(6)

        for idx, (tid, crop, team) in enumerate(player_data):
            tile = _PlayerTile(tid, crop, team)
            self._tiles.append(tile)
            row, col = divmod(idx, COLS)
            grid.addWidget(tile, row, col)

        scroll.setWidget(container)
        root.addWidget(scroll)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        confirm_btn = QPushButton("Bestätigen")
        confirm_btn.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        confirm_btn.clicked.connect(self.accept)
        btn_row.addWidget(confirm_btn)

        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Ergebnis
    # ------------------------------------------------------------------

    def get_corrections(self) -> Dict[int, int]:
        """Gibt {track_id: team_id} für alle geänderten Spieler zurück."""
        return {tile.track_id: tile.team for tile in self._tiles}
