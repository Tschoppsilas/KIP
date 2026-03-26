"""Phase 5 Tests: Teamzuordnung per HSV-Clustering.

Testet:
- extract_hsv_feature: Rückgabeform, Crop-Berechnung, Edge-Cases
- TeamAssigner.fit + predict: K-Means trennt klar verschiedene Farben
- TeamAssigner.get_team: Vorrang manuelle Korrekturen
- TeamAssigner.override / remove_override
- Should: Korrekturen bleiben erhalten (Clip-persistent)
- Could: save/load Persistenz
- Definition of Done: automatisch + manuell korrigierbar
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _red_frame(h=80, w=40) -> np.ndarray:
    """BGR-Frame, das komplett rot (BGR: 0, 0, 200) ist."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 2] = 200  # Rot im BGR-Raum
    return frame


def _blue_frame(h=80, w=40) -> np.ndarray:
    """BGR-Frame, das komplett blau (BGR: 200, 0, 0) ist."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = 200
    return frame


# ---------------------------------------------------------------------------
# extract_hsv_feature
# ---------------------------------------------------------------------------

class TestExtractHsvFeature(unittest.TestCase):

    def test_returns_array_shape(self):
        from src.tracking.team_assigner import extract_hsv_feature
        frame = _red_frame(80, 40)
        feat = extract_hsv_feature(frame, (0.0, 0.0, 40.0, 80.0))
        self.assertIsNotNone(feat)
        self.assertEqual(feat.shape, (3,))

    def test_red_and_blue_differ(self):
        """Rote und blaue Frames müssen sich deutlich im HSV unterscheiden."""
        from src.tracking.team_assigner import extract_hsv_feature
        fr = extract_hsv_feature(_red_frame(), (0, 0, 40, 80))
        fb = extract_hsv_feature(_blue_frame(), (0, 0, 40, 80))
        self.assertIsNotNone(fr)
        self.assertIsNotNone(fb)
        dist = float(np.linalg.norm(fr - fb))
        self.assertGreater(dist, 10.0)

    def test_zero_size_crop_returns_none(self):
        from src.tracking.team_assigner import extract_hsv_feature
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # x1==x2 → leerer Crop
        result = extract_hsv_feature(frame, (10.0, 10.0, 10.0, 30.0))
        self.assertIsNone(result)

    def test_uses_upper_half_only(self):
        """Die untere Hälfte (z.B. grün) darf das Ergebnis nicht dominieren."""
        from src.tracking.team_assigner import extract_hsv_feature
        frame = np.zeros((80, 40, 3), dtype=np.uint8)
        frame[:40, :, 2] = 200   # obere Hälfte: rot
        frame[40:, :, 1] = 200   # untere Hälfte: grün
        feat = extract_hsv_feature(frame, (0.0, 0.0, 40.0, 80.0))
        # H-Wert von Rot in HSV ≈ 0, von Grün ≈ 60; rot dominiert
        self.assertIsNotNone(feat)
        self.assertLess(feat[0], 30.0)  # Hue nahe 0 = rot


# ---------------------------------------------------------------------------
# TeamAssigner – Clustering
# ---------------------------------------------------------------------------

class TestTeamAssignerFit(unittest.TestCase):

    def _make_features(self):
        """Zwei klar getrennte Cluster: rot und blau."""
        from src.tracking.team_assigner import extract_hsv_feature
        red_feats = [
            extract_hsv_feature(_red_frame(), (0, 0, 40, 80)) for _ in range(5)
        ]
        blue_feats = [
            extract_hsv_feature(_blue_frame(), (0, 0, 40, 80)) for _ in range(5)
        ]
        return red_feats, blue_feats

    def test_fit_does_not_raise(self):
        from src.tracking.team_assigner import TeamAssigner
        r, b = self._make_features()
        assigner = TeamAssigner()
        assigner.fit(r + b)  # sollte nicht werfen

    def test_fit_requires_at_least_two_samples(self):
        from src.tracking.team_assigner import TeamAssigner
        assigner = TeamAssigner()
        with self.assertRaises(ValueError):
            assigner.fit([np.array([1, 2, 3], dtype=np.float32)])

    def test_predict_separates_colors(self):
        """Rote und blaue Spieler landen in verschiedenen Clustern."""
        from src.tracking.team_assigner import TeamAssigner, extract_hsv_feature
        r, b = self._make_features()
        assigner = TeamAssigner()
        assigner.fit(r + b)

        red_labels = {assigner.predict(f) for f in r}
        blue_labels = {assigner.predict(f) for f in b}
        # Alle roten in einem Cluster, alle blauen im anderen
        self.assertEqual(len(red_labels), 1)
        self.assertEqual(len(blue_labels), 1)
        self.assertNotEqual(red_labels, blue_labels)

    def test_predict_before_fit_raises(self):
        from src.tracking.team_assigner import TeamAssigner
        assigner = TeamAssigner()
        with self.assertRaises(RuntimeError):
            assigner.predict(np.array([0, 0, 0], dtype=np.float32))


# ---------------------------------------------------------------------------
# TeamAssigner – Manuelle Korrektur
# ---------------------------------------------------------------------------

class TestTeamAssignerOverride(unittest.TestCase):

    def _fitted_assigner(self):
        from src.tracking.team_assigner import TeamAssigner, extract_hsv_feature, TEAM_A, TEAM_B
        r = [extract_hsv_feature(_red_frame(), (0, 0, 40, 80)) for _ in range(3)]
        b = [extract_hsv_feature(_blue_frame(), (0, 0, 40, 80)) for _ in range(3)]
        a = TeamAssigner()
        a.fit(r + b)
        return a

    def test_override_takes_precedence(self):
        from src.tracking.team_assigner import TeamAssigner, TEAM_B, extract_hsv_feature
        assigner = self._fitted_assigner()
        red_feat = extract_hsv_feature(_red_frame(), (0, 0, 40, 80))
        auto_label = assigner.predict(red_feat)
        # Korrektur auf das andere Team
        other = 1 - auto_label
        assigner.override(track_id=99, team=other)
        self.assertEqual(assigner.get_team(99, red_feat), other)

    def test_override_persists_across_calls(self):
        """Korrektur bleibt für alle weiteren get_team()-Aufrufe erhalten."""
        from src.tracking.team_assigner import TeamAssigner, TEAM_A, extract_hsv_feature
        assigner = self._fitted_assigner()
        assigner.override(track_id=5, team=TEAM_A)
        for _ in range(3):
            self.assertEqual(assigner.get_team(5, np.zeros(3, dtype=np.float32)), TEAM_A)

    def test_remove_override(self):
        from src.tracking.team_assigner import TeamAssigner, TEAM_A, extract_hsv_feature
        assigner = self._fitted_assigner()
        red_feat = extract_hsv_feature(_red_frame(), (0, 0, 40, 80))
        auto = assigner.predict(red_feat)
        assigner.override(track_id=3, team=1 - auto)
        assigner.remove_override(track_id=3)
        # Nach Entfernen wieder automatisch
        self.assertEqual(assigner.get_team(3, red_feat), auto)

    def test_invalid_team_raises(self):
        from src.tracking.team_assigner import TeamAssigner
        assigner = TeamAssigner()
        with self.assertRaises(ValueError):
            assigner.override(track_id=1, team=99)

    def test_unknown_without_fit(self):
        from src.tracking.team_assigner import TeamAssigner, TEAM_UNKNOWN
        assigner = TeamAssigner()
        label = assigner.get_team(42, np.zeros(3, dtype=np.float32))
        self.assertEqual(label, TEAM_UNKNOWN)


# ---------------------------------------------------------------------------
# TeamAssigner – Persistenz (Could)
# ---------------------------------------------------------------------------

class TestTeamAssignerPersistence(unittest.TestCase):

    def _fitted_assigner(self):
        from src.tracking.team_assigner import TeamAssigner, TEAM_B, extract_hsv_feature
        r = [extract_hsv_feature(_red_frame(), (0, 0, 40, 80)) for _ in range(3)]
        b = [extract_hsv_feature(_blue_frame(), (0, 0, 40, 80)) for _ in range(3)]
        a = TeamAssigner()
        a.fit(r + b)
        a.override(track_id=7, team=TEAM_B)
        return a

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "assigner.json"
            self._fitted_assigner().save(p)
            self.assertTrue(p.exists())

    def test_roundtrip_overrides(self):
        from src.tracking.team_assigner import TeamAssigner, TEAM_B
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "assigner.json"
            a1 = self._fitted_assigner()
            a1.save(p)

            a2 = TeamAssigner()
            a2.load(p)
            self.assertEqual(a2.overrides.get(7), TEAM_B)

    def test_roundtrip_centers(self):
        from src.tracking.team_assigner import TeamAssigner, extract_hsv_feature
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "assigner.json"
            a1 = self._fitted_assigner()
            a1.save(p)

            a2 = TeamAssigner()
            a2.load(p)
            red_feat = extract_hsv_feature(_red_frame(), (0, 0, 40, 80))
            # Nach Laden sollte predict() wieder funktionieren
            result = a2.predict(red_feat)
            self.assertIn(result, (0, 1))

    def test_load_missing_file_raises(self):
        from src.tracking.team_assigner import TeamAssigner
        a = TeamAssigner()
        with self.assertRaises(FileNotFoundError):
            a.load("/nonexistent/path/assigner.json")


# ---------------------------------------------------------------------------
# Definition of Done: automatisch + manuell korrigierbar
# ---------------------------------------------------------------------------

class TestDefinitionOfDone(unittest.TestCase):

    def test_automatic_and_manual_correction(self):
        """Vollständiger Durchlauf: fit → predict → override → get_team."""
        from src.tracking.team_assigner import (
            TeamAssigner, extract_hsv_feature, TEAM_A, TEAM_B
        )
        red_feats = [extract_hsv_feature(_red_frame(), (0, 0, 40, 80)) for _ in range(4)]
        blue_feats = [extract_hsv_feature(_blue_frame(), (0, 0, 40, 80)) for _ in range(4)]

        assigner = TeamAssigner()
        assigner.fit(red_feats + blue_feats)

        # Automatische Zuordnung liefert konsistente Ergebnisse
        auto_red = {assigner.predict(f) for f in red_feats}
        auto_blue = {assigner.predict(f) for f in blue_feats}
        self.assertEqual(len(auto_red), 1)
        self.assertEqual(len(auto_blue), 1)
        self.assertNotEqual(auto_red, auto_blue)

        # Manuelle Korrektur: Spieler 10 zwingen auf Team B
        assigner.override(track_id=10, team=TEAM_B)
        self.assertEqual(assigner.get_team(10, red_feats[0]), TEAM_B)

        # Korrektur persistieren und neu laden
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "done.json"
            assigner.save(p)
            a2 = TeamAssigner()
            a2.load(p)
            self.assertEqual(a2.get_team(10, red_feats[0]), TEAM_B)


if __name__ == "__main__":
    unittest.main()
