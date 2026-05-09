"""
BoardWindow – Hauptfenster mit Live-Split-View (Video | Taktikboard).

Layout:
  Links:  aktuelles Videoframe mit Bounding-Boxes und Track-IDs (BGR → Qt)
  Rechts: interaktives TacticBoardWidget

Die Videohöhe wird an die native Board-Höhe angeglichen – dieselbe Logik wie
VideoExporter.write_frame(..., split_view=True).
"""

from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QShortcut,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QWidget,
)

from src.gui.pass_suggester import PassSuggestion
from src.gui.player_action_dialog import PlayerActionDialog
from src.gui.tactic_board_widget import DrawMode, TacticBoardWidget, TacticElement
from src.tracking.track_result import TrackResult
from src.utils import get_logger

logger = get_logger(__name__)

# BGR – konsistent mit src/utils/exporter.py render_board_frame
_TEAM_BOX_BGR = {
    0: (50, 50, 210),    # Team A – Rot
    1: (220, 100, 50),   # Team B – Blau
}

_MODE_LABELS = {
    DrawMode.SELECT:    "Werkzeug: Spieler auswählen  (Klick = auswählen / abwählen)",
    DrawMode.DRAW_PASS: "Werkzeug: Pass zeichnen  (1. Klick = Start · 2. Klick = Ziel)",
    DrawMode.DRAW_SHOT: "Werkzeug: Schuss zeichnen  (1. Klick = Start · 2. Klick = Ziel)",
    DrawMode.DRAW_PATH: "Werkzeug: Laufweg zeichnen  (Klicks = Punkte · Doppelklick = Fertig)",
}


def _annotate_video_with_boxes(
    frame_bgr: np.ndarray,
    track_result: TrackResult,
    teams: Dict[int, int],
) -> np.ndarray:
    """Zeichnet Bounding-Boxes und Track-IDs auf eine Kopie des Videoframes."""
    out = frame_bgr.copy()
    for i in range(len(track_result)):
        tid = track_result.track_ids[i]
        box = track_result.boxes[i].astype(np.int32)
        team = teams.get(tid, 0)
        color = _TEAM_BOX_BGR.get(team, (160, 160, 160))
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"id{tid}"
        cv2.putText(
            out, label, (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
    return out


class BoardWindow(QMainWindow):
    """Hauptfenster: Live-Video links, interaktives Taktikboard rechts."""

    def __init__(self, board_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("UniVision2Board – Video · Taktikboard")
        self.resize(1600, 820)

        self._paused: bool = False
        self.deleted_ids: Set[int] = set()
        self._assigner = None  # wird via set_assigner() gesetzt

        self._board = TacticBoardWidget(board_path)
        self._board.player_clicked.connect(self._on_player_clicked)

        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.setMinimumWidth(320)
        self._video_label.setStyleSheet("background-color: #12121c; color: #888;")
        self._video_label.setText("Video …")

        self._last_video_pixmap: Optional[QPixmap] = None

        split = QWidget()
        lay = QHBoxLayout(split)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)
        lay.addWidget(self._video_label, 1)
        lay.addWidget(self._board, 1)

        self.setCentralWidget(split)

        self._build_toolbar()
        self._build_statusbar()

        # Leertaste toggelt Pause (global im Fenster)
        shortcut = QShortcut(Qt.Key_Space, self)
        shortcut.activated.connect(self.toggle_pause)

    # ------------------------------------------------------------------
    # Öffentliche API – Steuerung
    # ------------------------------------------------------------------

    def set_assigner(self, assigner) -> None:
        """Setzt den TeamAssigner für Live-Korrekturen."""
        self._assigner = assigner

    def is_paused(self) -> bool:
        return self._paused

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._paused:
            self._pause_action.setChecked(True)
            self._pause_action.setText("▶  Weiter")
            self.setWindowTitle("UniVision2Board – Video · Taktikboard  [PAUSIERT]")
            self._status_lbl.setText("⏸  Pausiert — Klick auf Spieler für Aktion")
        else:
            self._pause_action.setChecked(False)
            self._pause_action.setText("⏸  Pause")
            self.setWindowTitle("UniVision2Board – Video · Taktikboard")
            self._status_lbl.setText(_MODE_LABELS.get(self._board._draw_mode, ""))

    # ------------------------------------------------------------------
    # Öffentliche API – Daten-Update
    # ------------------------------------------------------------------

    def update_frame(
        self,
        positions: Dict[int, Tuple[float, float]],
        teams: Dict[int, int],
        trajectories: Dict[int, List[Tuple[float, float]]],
        pass_suggestions: List[PassSuggestion],
    ) -> None:
        """Aktualisiert nur das Board (ohne Video-Panel)."""
        self._board.update_frame(positions, teams, trajectories, pass_suggestions)

    def update_live_split(
        self,
        video_bgr: np.ndarray,
        track_result: TrackResult,
        teams: Dict[int, int],
        positions: Dict[int, Tuple[float, float]],
        trajectories: Dict[int, List[Tuple[float, float]]],
        pass_suggestions: List[PassSuggestion],
    ) -> None:
        """
        Live-Split wie VideoExporter (split_view=True):
        Video auf Board-Höhe skalieren, Boxes zeichnen, rechts Board aktualisieren.
        """
        annotated = _annotate_video_with_boxes(video_bgr, track_result, teams)

        bh = self._board._board_h
        vh, vw = annotated.shape[:2]
        if vh != bh:
            annotated = cv2.resize(
                annotated, (int(vw * bh / vh), bh), interpolation=cv2.INTER_LINEAR
            )

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._last_video_pixmap = QPixmap.fromImage(qimg)
        self._scale_video_label()

        self._board.update_frame(positions, teams, trajectories, pass_suggestions)

    def _scale_video_label(self) -> None:
        if self._last_video_pixmap is None or self._last_video_pixmap.isNull():
            return
        self._video_label.setPixmap(
            self._last_video_pixmap.scaled(
                self._video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_video_label()

    def get_drawn_elements(self) -> List[TacticElement]:
        return self._board.get_drawn_elements()

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = QToolBar("Werkzeuge")
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        # Pause / Play
        self._pause_action = QAction("⏸  Pause", self, checkable=True)
        self._pause_action.setToolTip("Video pausieren / fortsetzen  (Leertaste)")
        self._pause_action.triggered.connect(self.toggle_pause)
        tb.addAction(self._pause_action)

        tb.addSeparator()

        # Werkzeug-Aktionen (exklusiv)
        self._action_select = QAction("⬡  Auswählen", self, checkable=True, checked=True)
        self._action_pass   = QAction("→  Pass",        self, checkable=True)
        self._action_shot   = QAction("⚡  Schuss",      self, checkable=True)
        self._action_path   = QAction("~  Laufweg",     self, checkable=True)

        for action, mode in [
            (self._action_select, DrawMode.SELECT),
            (self._action_pass,   DrawMode.DRAW_PASS),
            (self._action_shot,   DrawMode.DRAW_SHOT),
            (self._action_path,   DrawMode.DRAW_PATH),
        ]:
            action.triggered.connect(lambda checked, m=mode: self._set_mode(m))
            tb.addAction(action)

        tb.addSeparator()

        # Löschen-Button
        clear_action = QAction("🗑  Zeichnung löschen", self)
        clear_action.triggered.connect(self._board.clear_drawn_elements)
        tb.addAction(clear_action)

        tb.addSeparator()

        # Checkboxen für Einblendungen
        traj_cb = QCheckBox("Laufwege")
        traj_cb.setChecked(True)
        traj_cb.toggled.connect(self._board.set_show_trajectories)
        tb.addWidget(traj_cb)

        pass_cb = QCheckBox("Passvorschläge")
        pass_cb.setChecked(True)
        pass_cb.toggled.connect(self._board.set_show_pass_suggestions)
        tb.addWidget(pass_cb)

    def _set_mode(self, mode: DrawMode) -> None:
        self._board.set_draw_mode(mode)
        for action, m in [
            (self._action_select, DrawMode.SELECT),
            (self._action_pass,   DrawMode.DRAW_PASS),
            (self._action_shot,   DrawMode.DRAW_SHOT),
            (self._action_path,   DrawMode.DRAW_PATH),
        ]:
            action.setChecked(m == mode)
        self._status_lbl.setText(_MODE_LABELS.get(mode, ""))
        logger.info(f"Zeichenmodus: {mode.name}")

    # ------------------------------------------------------------------
    # Statusbar
    # ------------------------------------------------------------------

    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self._status_lbl = QLabel(_MODE_LABELS[DrawMode.SELECT])
        bar.addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_player_clicked(self, track_id: int) -> None:
        self._status_lbl.setText(f"Spieler #{track_id} ausgewählt.")
        logger.info(f"Spieler #{track_id} angeklickt.")

        current_team = 0
        if self._assigner is not None:
            tm = self._assigner.get_team(track_id)
            if tm is not None:
                current_team = tm

        dlg = PlayerActionDialog(track_id, current_team, parent=self)
        if dlg.exec_() != PlayerActionDialog.Accepted:
            self._board._selected_ids.discard(track_id)
            self._board.update()
            return

        action = dlg.get_action()
        if action in ("team_a", "team_b") and self._assigner is not None:
            new_team = 0 if action == "team_a" else 1
            self._assigner.correct(track_id, new_team)
            label = "A" if new_team == 0 else "B"
            self._status_lbl.setText(f"Spieler #{track_id} → Team {label}")
            logger.info(f"Spieler #{track_id} manuell zu Team {label} zugeordnet.")
        elif action == "delete":
            self.deleted_ids.add(track_id)
            self._board._selected_ids.discard(track_id)
            # Sofort aus der Anzeige entfernen
            self._board._player_positions.pop(track_id, None)
            self._board._player_teams.pop(track_id, None)
            self._board._trajectories.pop(track_id, None)
            self._board.update()
            self._status_lbl.setText(f"Spieler #{track_id} gelöscht (wird nicht mehr angezeigt).")
            logger.info(f"Spieler #{track_id} dauerhaft ausgeblendet.")
        else:
            self._board._selected_ids.discard(track_id)
            self._board.update()
