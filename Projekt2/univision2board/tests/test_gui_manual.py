"""
Manuelle GUI-Tests für TacticBoardWidget und BoardWindow.

ACHTUNG: Diese Tests erfordern ein echtes X11- oder Wayland-Display
und können NICHT headless ausgeführt werden.

Ausführen lokal:
    source .venv/bin/activate
    python -m pytest tests/test_gui_manual.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Taktikboard", "Taktikboard.png"
)

# Dieses Modul wird im normalen pytest-Lauf nicht gesammelt
collect_ignore_glob = ["test_gui_manual.py"]


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


class TestTacticBoardWidgetGUI:
    def test_widget_creates(self, qapp):
        if not os.path.exists(BOARD_PATH):
            pytest.skip("Taktikboard.png nicht gefunden.")
        from src.gui import TacticBoardWidget
        w = TacticBoardWidget(BOARD_PATH)
        assert w._board_w == 1280
        assert w._board_h == 720

    def test_update_frame_stores_state(self, qapp):
        if not os.path.exists(BOARD_PATH):
            pytest.skip("Taktikboard.png nicht gefunden.")
        from src.gui import TacticBoardWidget
        from src.gui.pass_suggester import PassSuggestion
        w = TacticBoardWidget(BOARD_PATH)
        positions = {1: (200.0, 300.0), 2: (600.0, 400.0)}
        teams = {1: 0, 2: 1}
        trajectories = {1: [(150.0, 280.0), (200.0, 300.0)]}
        suggestions = [PassSuggestion(1, 2, (200.0, 300.0), (600.0, 400.0))]
        w.update_frame(positions, teams, trajectories, suggestions)
        assert w._player_positions == positions
        assert w._player_teams == teams

    def test_draw_mode_switch(self, qapp):
        if not os.path.exists(BOARD_PATH):
            pytest.skip("Taktikboard.png nicht gefunden.")
        from src.gui import TacticBoardWidget, DrawMode
        w = TacticBoardWidget(BOARD_PATH)
        w.set_draw_mode(DrawMode.DRAW_PASS)
        assert w._draw_mode == DrawMode.DRAW_PASS

    def test_board_to_widget_roundtrip(self, qapp):
        if not os.path.exists(BOARD_PATH):
            pytest.skip("Taktikboard.png nicht gefunden.")
        from src.gui import TacticBoardWidget
        w = TacticBoardWidget(BOARD_PATH)
        w.resize(1280, 720)
        bx, by = 640.0, 360.0
        wx, wy = w._board_to_widget(bx, by)
        bx2, by2 = w._widget_to_board(wx, wy)
        assert abs(bx2 - bx) < 1.0
        assert abs(by2 - by) < 1.0

    def test_board_window_creates(self, qapp):
        if not os.path.exists(BOARD_PATH):
            pytest.skip("Taktikboard.png nicht gefunden.")
        from src.gui import BoardWindow
        win = BoardWindow(BOARD_PATH)
        assert win.windowTitle() == "UniVision2Board – Video · Taktikboard"
