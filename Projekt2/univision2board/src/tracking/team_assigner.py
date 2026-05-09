"""
Automatische Teamzuordnung via HSV-Farbmerkmale und K-Means-Clustering.

Ablauf:
1. Pro Frame werden die Trikot-Farben (obere Hälfte der Bounding-Box)
   aller getrackten Spieler in HSV extrahiert.
2. Nach Akkumulation über mehrere Frames wird pro Track-ID die mittlere
   HSV-Farbe berechnet.
3. K-Means (k=2) gruppiert alle Spieler in Team A (0) und Team B (1).
4. Manuelle Korrekturen des Trainers werden gespeichert und überschreiben
   die automatische Zuweisung dauerhaft.
"""

import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from src.tracking.track_result import TrackResult
from src.utils import get_logger

logger = get_logger(__name__)

TEAM_A = 0
TEAM_B = 1
TEAM_NAMES = {TEAM_A: "Team A", TEAM_B: "Team B"}


class TeamAssigner:
    """
    Weist Spieler-Track-IDs automatisch einem von zwei Teams zu.

    Parameters:
        min_samples: Mindestanzahl Farbproben pro Spieler vor dem Clustering.
    """

    def __init__(self, min_samples: int = 3):
        self.min_samples = min_samples
        self._color_samples: Dict[int, List[np.ndarray]] = {}
        self._assignments: Dict[int, int] = {}
        self._corrections: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Farbproben sammeln
    # ------------------------------------------------------------------

    def add_frame(self, frame: np.ndarray, track_result: TrackResult) -> None:
        """Extrahiert Trikot-Farben aller Spieler im aktuellen Frame."""
        for i in range(len(track_result)):
            tid = track_result.track_ids[i]
            box = track_result.boxes[i]
            color = self._extract_shirt_color(frame, box)
            if color is not None:
                self._color_samples.setdefault(tid, []).append(color)

    def _extract_shirt_color(
        self, frame: np.ndarray, box: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Mittlerer HSV-Wert der Trikot-Region (obere 50 % der Bounding-Box).

        Die obere Hälfte enthält in der Regel das Trikot und vermeidet
        Shorts/Boden, die das Clustering verfälschen würden.
        """
        x1, y1, x2, y2 = map(int, box)
        h = y2 - y1
        w = x2 - x1
        if h < 4 or w < 4:
            return None

        shirt_y2 = y1 + max(1, h // 2)
        crop = frame[y1:shirt_y2, x1:x2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Teamzuordnung berechnen
    # ------------------------------------------------------------------

    def assign_teams(self) -> Dict[int, int]:
        """
        Führt K-Means (k=2) über die akkumulierten Farbproben durch
        und gibt {track_id: team_id} zurück.

        Spieler mit zu wenigen Proben erhalten provisorisch Team A.
        Manuelle Korrekturen überschreiben das Ergebnis.
        """
        ids, features = [], []
        for tid, samples in self._color_samples.items():
            if len(samples) >= self.min_samples:
                ids.append(tid)
                features.append(np.mean(samples, axis=0))

        if len(features) < 2:
            logger.warning(
                f"Zu wenige Spieler für K-Means ({len(features)}) – "
                "alle Spieler werden vorläufig Team A zugeordnet."
            )
            self._assignments = {tid: TEAM_A for tid in self._color_samples}
        else:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
            labels = kmeans.fit_predict(np.array(features))
            self._assignments = {
                tid: int(label) for tid, label in zip(ids, labels)
            }
            logger.info(
                f"K-Means abgeschlossen: {len(ids)} Spieler zugeordnet "
                f"(Team A: {sum(l == TEAM_A for l in labels)}, "
                f"Team B: {sum(l == TEAM_B for l in labels)})."
            )

        # Spieler mit zu wenigen Proben erhalten Team A als Fallback
        for tid in self._color_samples:
            if tid not in self._assignments:
                self._assignments[tid] = TEAM_A

        # Manuelle Korrekturen haben immer Vorrang
        self._assignments.update(self._corrections)
        return dict(self._assignments)

    # ------------------------------------------------------------------
    # Abfrage und Korrektur
    # ------------------------------------------------------------------

    def get_team(self, track_id: int) -> Optional[int]:
        """Gibt Team-ID (0=A, 1=B) zurück oder None wenn unbekannt."""
        return self._assignments.get(track_id)

    def get_all_assignments(self) -> Dict[int, int]:
        return dict(self._assignments)

    def correct(self, track_id: int, team_id: int) -> None:
        """Manuelle Korrektur: überschreibt die automatische Zuweisung dauerhaft."""
        if team_id not in (TEAM_A, TEAM_B):
            raise ValueError(f"Ungültige Team-ID: {team_id} – erlaubt: 0 (A) oder 1 (B).")
        self._corrections[track_id] = team_id
        self._assignments[track_id] = team_id
        logger.info(f"Manuelle Korrektur: Spieler {track_id} → {TEAM_NAMES[team_id]}")

    def get_mean_color(self, track_id: int) -> Optional[np.ndarray]:
        """Mittlere HSV-Farbe eines Spielers (für Darstellung im UI)."""
        samples = self._color_samples.get(track_id)
        if not samples:
            return None
        return np.mean(samples, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistenz
    # ------------------------------------------------------------------

    def save_corrections(self, path: str) -> None:
        """Speichert manuelle Korrekturen als JSON (int-Schlüssel als Strings)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self._corrections.items()}, f, indent=2)
        logger.info(f"Team-Korrekturen gespeichert: {path}")

    def load_corrections(self, path: str) -> None:
        """Lädt manuelle Korrekturen aus JSON und wendet sie sofort an."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._corrections = {int(k): v for k, v in data.items()}
        self._assignments.update(self._corrections)
        logger.info(f"Team-Korrekturen geladen: {path} ({len(self._corrections)} Einträge)")
