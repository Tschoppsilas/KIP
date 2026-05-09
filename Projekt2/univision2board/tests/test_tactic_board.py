"""Tests für PassSuggester, TacticElement-Logik und Koordinatentransformation."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Taktikboard", "Taktikboard.png"
)

# ---------------------------------------------------------------------------
# Hilfsfunktion: Prüft ob ein Display vorhanden ist (für GUI-Tests)
# ---------------------------------------------------------------------------
def _has_display() -> bool:
    """Gibt True zurück wenn ein X- oder Wayland-Display verfügbar ist."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# ---------------------------------------------------------------------------
# PassSuggester
# ---------------------------------------------------------------------------

class TestPassSuggester:
    def _make(self, **kwargs):
        from src.gui.pass_suggester import PassSuggester
        return PassSuggester(**kwargs)

    def test_suggests_pass_between_teammates(self):
        sug = self._make(max_pass_dist=500, block_radius=30)
        positions = {1: (100.0, 100.0), 2: (300.0, 100.0)}
        teams = {1: 0, 2: 0}
        result = sug.suggest(positions, teams)
        assert len(result) == 1
        assert {result[0].from_id, result[0].to_id} == {1, 2}

    def test_no_pass_between_opponents(self):
        sug = self._make(max_pass_dist=500)
        positions = {1: (100.0, 100.0), 2: (300.0, 100.0)}
        teams = {1: 0, 2: 1}
        result = sug.suggest(positions, teams)
        assert len(result) == 0

    def test_pass_blocked_by_opponent(self):
        sug = self._make(max_pass_dist=600, block_radius=30)
        positions = {
            1: (100.0, 100.0),
            2: (500.0, 100.0),
            3: (300.0, 100.0),  # Gegner direkt auf der Linie
        }
        teams = {1: 0, 2: 0, 3: 1}
        result = sug.suggest(positions, teams)
        assert len(result) == 0

    def test_pass_not_blocked_when_opponent_off_line(self):
        sug = self._make(max_pass_dist=600, block_radius=30)
        positions = {
            1: (100.0, 100.0),
            2: (500.0, 100.0),
            3: (300.0, 300.0),  # Gegner weit daneben
        }
        teams = {1: 0, 2: 0, 3: 1}
        result = sug.suggest(positions, teams)
        assert len(result) == 1

    def test_distance_limit(self):
        sug = self._make(max_pass_dist=100)
        positions = {1: (0.0, 0.0), 2: (200.0, 0.0)}
        teams = {1: 0, 2: 0}
        result = sug.suggest(positions, teams)
        assert len(result) == 0

    def test_max_per_player_limit(self):
        sug = self._make(max_pass_dist=1000, max_per_player=1)
        positions = {1: (100.0, 100.0), 2: (200.0, 100.0), 3: (300.0, 100.0)}
        teams = {1: 0, 2: 0, 3: 0}
        result = sug.suggest(positions, teams)
        from_1 = [s for s in result if s.from_id == 1]
        assert len(from_1) <= 1

    def test_no_duplicate_pairs(self):
        sug = self._make(max_pass_dist=1000)
        positions = {1: (100.0, 100.0), 2: (200.0, 100.0)}
        teams = {1: 0, 2: 0}
        result = sug.suggest(positions, teams)
        pairs = [frozenset([s.from_id, s.to_id]) for s in result]
        assert len(pairs) == len(set(pairs))

    def test_empty_input(self):
        sug = self._make()
        assert sug.suggest({}, {}) == []

    def test_suggestion_attributes(self):
        sug = self._make(max_pass_dist=1000)
        positions = {1: (100.0, 200.0), 2: (400.0, 200.0)}
        teams = {1: 0, 2: 0}
        result = sug.suggest(positions, teams)
        assert len(result) == 1
        s = result[0]
        assert s.from_pos == (100.0, 200.0)
        assert s.to_pos == (400.0, 200.0)


# ---------------------------------------------------------------------------
# TacticElement – reine Datenlogik (kein Display benötigt)
# ---------------------------------------------------------------------------

class TestTacticElement:
    def test_create_pass_element(self):
        from src.gui.tactic_board_widget import TacticElement
        el = TacticElement(kind="pass", points=[(10.0, 20.0), (300.0, 400.0)])
        assert el.kind == "pass"
        assert len(el.points) == 2

    def test_create_shot_element(self):
        from src.gui.tactic_board_widget import TacticElement
        el = TacticElement(kind="shot", points=[(50.0, 50.0), (600.0, 200.0)])
        assert el.kind == "shot"

    def test_create_path_element(self):
        from src.gui.tactic_board_widget import TacticElement
        el = TacticElement(kind="path", points=[(0.0, 0.0), (100.0, 50.0), (200.0, 80.0)])
        assert el.kind == "path"
        assert len(el.points) == 3

    def test_draw_mode_enum(self):
        from src.gui.tactic_board_widget import DrawMode
        assert DrawMode.SELECT is not DrawMode.DRAW_PASS
        assert DrawMode.DRAW_PASS is not DrawMode.DRAW_SHOT
        assert DrawMode.DRAW_SHOT is not DrawMode.DRAW_PATH


# ---------------------------------------------------------------------------
# Koordinatentransformation – als reine Mathematik testbar
# ---------------------------------------------------------------------------

class TestCoordinateTransform:
    """
    Testet die Board↔Widget-Koordinatentransformation als isolierte Funktion.
    Kein QWidget, kein QPixmap nötig.
    """

    @staticmethod
    def _transform(board_w, board_h, widget_w, widget_h, bx, by):
        """Simuliert _board_to_widget: KeepAspectRatio-Skalierung."""
        scale = min(widget_w / board_w, widget_h / board_h)
        disp_w = board_w * scale
        disp_h = board_h * scale
        ox = (widget_w - disp_w) / 2
        oy = (widget_h - disp_h) / 2
        return bx * scale + ox, by * scale + oy

    @staticmethod
    def _inv_transform(board_w, board_h, widget_w, widget_h, wx, wy):
        """Simuliert _widget_to_board."""
        scale = min(widget_w / board_w, widget_h / board_h)
        disp_w = board_w * scale
        disp_h = board_h * scale
        ox = (widget_w - disp_w) / 2
        oy = (widget_h - disp_h) / 2
        return (wx - ox) / scale, (wy - oy) / scale

    def test_roundtrip_center(self):
        bx, by = 640.0, 360.0
        wx, wy = self._transform(1280, 720, 1280, 720, bx, by)
        bx2, by2 = self._inv_transform(1280, 720, 1280, 720, wx, wy)
        assert abs(bx2 - bx) < 0.01
        assert abs(by2 - by) < 0.01

    def test_roundtrip_corner(self):
        bx, by = 0.0, 0.0
        wx, wy = self._transform(1280, 720, 800, 600, bx, by)
        bx2, by2 = self._inv_transform(1280, 720, 800, 600, wx, wy)
        assert abs(bx2 - bx) < 0.01
        assert abs(by2 - by) < 0.01

    def test_aspect_ratio_preserved(self):
        """Bei abweichendem Widget-Format muss KeepAspectRatio Letterboxing erzeugen."""
        # Widget 800x600, Board 1280x720 → scale = min(800/1280, 600/720) = 0.625/0.833 = 0.625
        scale = min(800 / 1280, 600 / 720)
        assert abs(scale - 0.625) < 0.001
        # Offset-Y (Letterboxing oben/unten) muss > 0 sein
        disp_h = 720 * scale
        oy = (600 - disp_h) / 2
        assert oy > 0


# ---------------------------------------------------------------------------
# Hinweis: GUI-Widget-Tests (TacticBoardWidget, BoardWindow)
# ---------------------------------------------------------------------------
# Die vollständigen Widget-Tests sind in tests/test_gui_manual.py ausgelagert.
# Sie erfordern ein echtes X- oder Wayland-Display und können auf diesem
# Server nicht headless ausgeführt werden.
# Ausführung lokal: python -m pytest tests/test_gui_manual.py -v
