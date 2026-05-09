"""Tests für TeamAssigner (Farbextraktion, K-Means, Korrekturen)."""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "finetune", "runs", "train", "weights", "best.pt"
)
VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Videos", "Muenchenstein_1.mp4"
)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_track_result(track_ids, boxes):
    from src.tracking import TrackResult
    tr = TrackResult(frame_idx=0)
    tr.track_ids = list(track_ids)
    tr.boxes = [np.array(b, dtype=np.float32) for b in boxes]
    tr.confidences = [0.9] * len(track_ids)
    return tr


def _solid_frame(color_bgr, h=200, w=200):
    """Einfarbiges Testbild."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color_bgr
    return frame


# ---------------------------------------------------------------------------
# TeamAssigner – Einheit
# ---------------------------------------------------------------------------

class TestTeamAssigner:
    def test_add_frame_accumulates_samples(self):
        from src.tracking import TeamAssigner
        ta = TeamAssigner(min_samples=1)
        frame = _solid_frame((0, 0, 200))  # Rot-ish in BGR
        tr = _make_track_result([1, 2], [(0, 0, 50, 80), (100, 0, 150, 80)])
        ta.add_frame(frame, tr)
        assert 1 in ta._color_samples
        assert 2 in ta._color_samples
        assert len(ta._color_samples[1]) == 1

    def test_assign_two_teams_from_distinct_colors(self):
        """K-Means muss zwei klar verschiedene Farben (rot vs. blau) trennen."""
        from src.tracking import TeamAssigner, TEAM_A, TEAM_B

        ta = TeamAssigner(min_samples=1)

        # Spieler 1–5: rote Trikots  (BGR 0,0,200)
        # Spieler 6–10: blaue Trikots (BGR 200,0,0)
        red_frame = _solid_frame((0, 0, 200))
        blue_frame = _solid_frame((200, 0, 0))

        for tid in range(1, 6):
            tr = _make_track_result([tid], [(10, 10, 60, 60)])
            for _ in range(3):
                ta.add_frame(red_frame, tr)

        for tid in range(6, 11):
            tr = _make_track_result([tid], [(10, 10, 60, 60)])
            for _ in range(3):
                ta.add_frame(blue_frame, tr)

        assignments = ta.assign_teams()
        assert len(assignments) == 10

        red_teams = {assignments[tid] for tid in range(1, 6)}
        blue_teams = {assignments[tid] for tid in range(6, 11)}

        # Alle roten Spieler müssen im gleichen Team sein, alle blauen im anderen
        assert len(red_teams) == 1, "Rote Spieler wurden verschiedenen Teams zugeordnet."
        assert len(blue_teams) == 1, "Blaue Spieler wurden verschiedenen Teams zugeordnet."
        assert red_teams != blue_teams, "Rote und blaue Spieler landeten im gleichen Team."

    def test_manual_correction_overrides_kmeans(self):
        from src.tracking import TeamAssigner, TEAM_A, TEAM_B

        ta = TeamAssigner(min_samples=1)
        red_frame = _solid_frame((0, 0, 200))
        blue_frame = _solid_frame((200, 0, 0))

        for tid in range(1, 4):
            tr = _make_track_result([tid], [(10, 10, 60, 60)])
            ta.add_frame(red_frame, tr)
        for tid in range(4, 7):
            tr = _make_track_result([tid], [(10, 10, 60, 60)])
            ta.add_frame(blue_frame, tr)

        ta.assign_teams()
        original_team = ta.get_team(1)
        other_team = TEAM_B if original_team == TEAM_A else TEAM_A

        ta.correct(1, other_team)
        assert ta.get_team(1) == other_team, "Manuelle Korrektur wurde nicht übernommen."

    def test_correction_invalid_team_raises(self):
        from src.tracking import TeamAssigner
        ta = TeamAssigner()
        with pytest.raises(ValueError):
            ta.correct(1, 99)

    def test_save_and_load_corrections(self):
        from src.tracking import TeamAssigner, TEAM_B

        ta = TeamAssigner(min_samples=1)
        ta._color_samples = {1: [np.array([10, 200, 180])]}
        ta.assign_teams()
        ta.correct(1, TEAM_B)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            ta.save_corrections(tmp)

            ta2 = TeamAssigner()
            ta2._assignments = {}
            ta2.load_corrections(tmp)
            assert ta2._corrections[1] == TEAM_B
        finally:
            os.unlink(tmp)

    def test_get_mean_color(self):
        from src.tracking import TeamAssigner
        ta = TeamAssigner(min_samples=1)
        frame = _solid_frame((100, 150, 200))
        tr = _make_track_result([7], [(10, 10, 80, 80)])
        ta.add_frame(frame, tr)
        color = ta.get_mean_color(7)
        assert color is not None
        assert color.shape == (3,)

    def test_too_few_players_fallback(self):
        """Bei nur einem Spieler darf kein Fehler auftreten."""
        from src.tracking import TeamAssigner, TEAM_A
        ta = TeamAssigner(min_samples=1)
        frame = _solid_frame((0, 200, 0))
        tr = _make_track_result([1], [(10, 10, 60, 60)])
        ta.add_frame(frame, tr)
        assignments = ta.assign_teams()
        assert assignments[1] == TEAM_A


# ---------------------------------------------------------------------------
# Integration: echtes Video + Tracking + Teamzuordnung
# ---------------------------------------------------------------------------

class TestTeamAssignerIntegration:
    def test_team_split_on_real_video(self):
        """
        Auf dem echten Video müssen nach 30 Frames zwei Teams erkennbar sein –
        d.h. beide Team-IDs (0 und 1) sollen vorkommen.
        """
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VIDEO_PATH):
            pytest.skip("Modell oder Video nicht gefunden.")
        import cv2
        from src.tracking import PlayerTracker, TeamAssigner

        tracker = PlayerTracker(MODEL_PATH, conf=0.25)
        assigner = TeamAssigner(min_samples=3)

        cap = cv2.VideoCapture(VIDEO_PATH)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)

        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            track_result = tracker.update(frame, frame_idx=i)
            assigner.add_frame(frame, track_result)

        cap.release()

        assignments = assigner.assign_teams()
        assert len(assignments) > 0, "Keine Spieler erkannt."

        teams_found = set(assignments.values())
        assert len(teams_found) == 2, (
            f"Erwartet 2 Teams, gefunden: {teams_found}. "
            "Evtl. zu wenige Spieler im Bildausschnitt?"
        )
        print(f"\n[Teamzuordnung] {len(assignments)} Spieler → Teams: {teams_found}")
