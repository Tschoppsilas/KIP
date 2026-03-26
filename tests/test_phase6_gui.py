"""Phase 6 Tests: Taktikboard-Renderer (kein Display erforderlich).

Testet:
- BoardRenderer: Koordinatentransformation (board_to_canvas / canvas_to_board)
- BoardRenderer.render(): Gibt ein PIL-Image in korrekter Grösse zurück
- PlayerSymbol: Standardlabel, Klassen
- Arrow: Typen und Felder
- BoardState: Erstellung und Zusammenführung
- Passvorschlag-Logik (regelbasiert)
"""

import unittest
from pathlib import Path


BOARD_IMG = Path(__file__).parent.parent / "Taktikboard" / "Taktikboard.png"


class TestBoardRenderer(unittest.TestCase):

    def _make_renderer(self):
        from src.gui.board_renderer import BoardRenderer
        return BoardRenderer(BOARD_IMG, field_width_m=40.0, field_height_m=20.0)

    def test_canvas_size(self):
        r = self._make_renderer()
        self.assertEqual(r.canvas_w, 1280)
        self.assertEqual(r.canvas_h, 720)

    def test_board_to_canvas_origin(self):
        """Feldursprung (0,0) liegt links-oben im Padding-Bereich."""
        r = self._make_renderer()
        cx, cy = r.board_to_canvas(0.0, 0.0)
        self.assertGreater(cx, 0)
        self.assertGreater(cy, 0)
        self.assertLess(cx, r.canvas_w // 2)
        self.assertLess(cy, r.canvas_h // 2)

    def test_board_to_canvas_far_corner(self):
        """Feldecke (40,20) liegt rechts-unten im Padding-Bereich."""
        r = self._make_renderer()
        cx, cy = r.board_to_canvas(40.0, 20.0)
        self.assertGreater(cx, r.canvas_w // 2)
        self.assertGreater(cy, r.canvas_h // 2)
        self.assertLess(cx, r.canvas_w)
        self.assertLess(cy, r.canvas_h)

    def test_roundtrip_coordinates(self):
        """board_to_canvas → canvas_to_board sollte den Ausgangswert annähern."""
        r = self._make_renderer()
        for bx, by in [(0.0, 0.0), (20.0, 10.0), (40.0, 20.0), (5.5, 12.3)]:
            cx, cy = r.board_to_canvas(bx, by)
            bx2, by2 = r.canvas_to_board(cx, cy)
            self.assertAlmostEqual(bx, bx2, delta=0.5)
            self.assertAlmostEqual(by, by2, delta=0.5)

    def test_render_returns_correct_size(self):
        from src.gui.board_renderer import BoardRenderer, BoardState
        r = self._make_renderer()
        img = r.render(BoardState())
        self.assertEqual(img.size, (1280, 720))

    def test_render_rgb_mode(self):
        from src.gui.board_renderer import BoardRenderer, BoardState
        r = self._make_renderer()
        img = r.render_rgb(BoardState())
        self.assertEqual(img.mode, "RGB")

    def test_render_with_player(self):
        """Render mit einem Spieler darf nicht werfen."""
        from src.gui.board_renderer import BoardRenderer, BoardState, PlayerSymbol
        r = self._make_renderer()
        state = BoardState(players=[
            PlayerSymbol(track_id=3, team=0, class_name="player",
                         board_x=20.0, board_y=10.0),
        ])
        img = r.render(state)
        self.assertEqual(img.size, (1280, 720))

    def test_render_with_arrow(self):
        """Render mit einem Pfeil darf nicht werfen."""
        from src.gui.board_renderer import BoardRenderer, BoardState, Arrow
        r = self._make_renderer()
        state = BoardState(arrows=[Arrow(5.0, 5.0, 15.0, 10.0, kind="pass")])
        img = r.render(state)
        self.assertIsNotNone(img)

    def test_render_all_arrow_types(self):
        from src.gui.board_renderer import BoardRenderer, BoardState, Arrow
        r = self._make_renderer()
        for kind in ("pass", "shot", "run"):
            state = BoardState(arrows=[Arrow(0, 0, 10, 10, kind=kind)])
            img = r.render(state)
            self.assertIsNotNone(img)

    def test_render_goalkeeper_symbol(self):
        from src.gui.board_renderer import BoardRenderer, BoardState, PlayerSymbol
        r = self._make_renderer()
        state = BoardState(players=[
            PlayerSymbol(1, team=0, class_name="goalkeeper", board_x=2.0, board_y=10.0),
        ])
        r.render(state)  # darf nicht werfen

    def test_render_ball_symbol(self):
        from src.gui.board_renderer import BoardRenderer, BoardState, PlayerSymbol
        r = self._make_renderer()
        state = BoardState(players=[
            PlayerSymbol(0, team=-1, class_name="ball", board_x=20.0, board_y=10.0),
        ])
        r.render(state)


class TestPlayerSymbol(unittest.TestCase):

    def test_default_label_is_track_id(self):
        from src.gui.board_renderer import PlayerSymbol
        p = PlayerSymbol(track_id=7, team=0, class_name="player",
                         board_x=10.0, board_y=5.0)
        self.assertEqual(p.label, "7")

    def test_custom_label(self):
        from src.gui.board_renderer import PlayerSymbol
        p = PlayerSymbol(track_id=1, team=1, class_name="goalkeeper",
                         board_x=0.0, board_y=10.0, label="TW")
        self.assertEqual(p.label, "TW")


class TestArrow(unittest.TestCase):

    def test_default_kind(self):
        from src.gui.board_renderer import Arrow
        a = Arrow(0, 0, 10, 5)
        self.assertEqual(a.kind, "pass")

    def test_shot_kind(self):
        from src.gui.board_renderer import Arrow
        a = Arrow(0, 0, 5, 10, kind="shot")
        self.assertEqual(a.kind, "shot")


class TestBoardState(unittest.TestCase):

    def test_empty_state(self):
        from src.gui.board_renderer import BoardState
        s = BoardState()
        self.assertEqual(s.players, [])
        self.assertEqual(s.arrows, [])
        self.assertEqual(s.frame_index, -1)

    def test_state_with_data(self):
        from src.gui.board_renderer import BoardState, PlayerSymbol, Arrow
        s = BoardState(
            players=[PlayerSymbol(1, 0, "player", 5.0, 5.0)],
            arrows=[Arrow(0, 0, 10, 10)],
            frame_index=42,
        )
        self.assertEqual(len(s.players), 1)
        self.assertEqual(len(s.arrows), 1)
        self.assertEqual(s.frame_index, 42)


class TestPassSuggestions(unittest.TestCase):
    """Regelbasierte Passvorschläge: Spieler desselben Teams in Reichweite."""

    def _make_state(self, players):
        from src.gui.board_renderer import BoardState
        return BoardState(players=players)

    def test_same_team_close_gets_arrow(self):
        from src.gui.board_renderer import PlayerSymbol, Arrow
        p1 = PlayerSymbol(1, team=0, class_name="player", board_x=10.0, board_y=10.0)
        p2 = PlayerSymbol(2, team=0, class_name="player", board_x=15.0, board_y=10.0)
        state = self._make_state([p1, p2])

        import math
        max_dist = 8.0
        added: set = set()
        players = [p for p in state.players if p.class_name != "ball"]
        for i, a in enumerate(players):
            for j, b in enumerate(players):
                if i >= j or a.team != b.team or a.team == -1:
                    continue
                dist = math.hypot(a.board_x - b.board_x, a.board_y - b.board_y)
                if dist <= max_dist and (i, j) not in added:
                    state.arrows.append(Arrow(a.board_x, a.board_y,
                                              b.board_x, b.board_y, kind="pass"))
                    added.add((i, j))

        self.assertEqual(len(state.arrows), 1)
        self.assertEqual(state.arrows[0].kind, "pass")

    def test_different_teams_no_arrow(self):
        from src.gui.board_renderer import PlayerSymbol, Arrow
        p1 = PlayerSymbol(1, team=0, class_name="player", board_x=10.0, board_y=10.0)
        p2 = PlayerSymbol(2, team=1, class_name="player", board_x=12.0, board_y=10.0)
        state = self._make_state([p1, p2])
        # Keine Pfeile für verschiedene Teams
        arrows_before = len(state.arrows)
        import math
        for i, a in enumerate(state.players):
            for j, b in enumerate(state.players):
                if i >= j or a.team != b.team:
                    continue
                dist = math.hypot(a.board_x - b.board_x, a.board_y - b.board_y)
                if dist <= 8.0:
                    state.arrows.append(Arrow(a.board_x, a.board_y, b.board_x, b.board_y))
        self.assertEqual(len(state.arrows), arrows_before)

    def test_too_far_no_arrow(self):
        from src.gui.board_renderer import PlayerSymbol, Arrow
        p1 = PlayerSymbol(1, team=0, class_name="player", board_x=0.0, board_y=0.0)
        p2 = PlayerSymbol(2, team=0, class_name="player", board_x=30.0, board_y=0.0)
        state = self._make_state([p1, p2])
        import math
        for i, a in enumerate(state.players):
            for j, b in enumerate(state.players):
                if i >= j or a.team != b.team:
                    continue
                dist = math.hypot(a.board_x - b.board_x, a.board_y - b.board_y)
                if dist <= 8.0:
                    state.arrows.append(Arrow(a.board_x, a.board_y, b.board_x, b.board_y))
        self.assertEqual(len(state.arrows), 0)


if __name__ == "__main__":
    unittest.main()
