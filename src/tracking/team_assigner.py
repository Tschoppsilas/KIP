"""Phase 5: Automatische Teamzuordnung per HSV-Farbclustering.

Ablauf:
1. Obere Hälfte der Spieler-Bounding-Box wird ausgeschnitten (Trikot, kein Boden).
2. Mittlere HSV-Werte (Histogramm-Modus) werden als Merkmal verwendet.
3. K-Means (k=2) trennt Team A von Team B.
4. Manuelle Korrekturen überschreiben die automatische Zuordnung und bleiben
   für den gesamten Clip erhalten.
5. Optional: Zuordnung in/aus JSON laden für Wiederverwendung.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Öffentliche Labels
TEAM_A = 0
TEAM_B = 1
TEAM_UNKNOWN = -1


def extract_hsv_feature(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """Extrahiert einen HSV-Farbvektor aus der oberen Trikothälfte einer BBox.

    Die obere 50 % der Box enthält meistens das Trikot und vermeidet den Boden
    sowie Hosenbeine.  Zurückgegeben wird ein Vektor [H_mean, S_mean, V_mean].

    Args:
        frame: BGR-Frame als NumPy-Array (H×W×3).
        bbox:  (x1, y1, x2, y2) in Pixelkoordinaten.

    Returns:
        Numpy-Array mit Form (3,) oder ``None`` bei ungültiger Crop-Größe.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h = y2 - y1
    # Nur obere 50 % → Trikot
    y_mid = y1 + h // 2
    crop = frame[y1:y_mid, x1:x2]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Mittelwert über den Crop
    return hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)


class TeamAssigner:
    """Ordnet Spieler automatisch einem von zwei Teams zu (K-Means, k=2).

    Manuelle Korrekturen (``override``) werden pro ``track_id`` gespeichert
    und im gesamten Clip angewendet.  Sie können optional in eine JSON-Datei
    persistiert werden (Could-Kriterium).

    Verwendung::

        assigner = TeamAssigner()
        # Erst alle Spieler eines Frames sammeln, dann fitten:
        assigner.fit(features)          # list[np.ndarray] mit HSV-Vektoren
        label = assigner.predict(feat)  # TEAM_A oder TEAM_B

        # Manuelle Korrektur:
        assigner.override(track_id=7, team=TEAM_B)
        label = assigner.get_team(track_id=7, feature=feat)  # → TEAM_B
    """

    def __init__(self) -> None:
        self._centers: np.ndarray | None = None   # (2, 3) cluster centres
        self._overrides: dict[int, int] = {}       # track_id → TEAM_A/B
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def fit(self, features: list[np.ndarray]) -> None:
        """Führt K-Means (k=2) auf den gegebenen HSV-Merkmalsvektoren durch.

        Args:
            features: Liste von HSV-Merkmalsvektoren (je Shape (3,)).
                      Mindestens 2 Einträge erforderlich.

        Raises:
            ValueError: Falls ``features`` weniger als 2 Einträge enthält.
        """
        if len(features) < 2:
            raise ValueError("Mindestens 2 Feature-Vektoren für K-Means benötigt.")

        data = np.stack(features, axis=0).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            data, 2, None, criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS
        )
        self._centers = centers          # (2, 3)
        self._fitted = True
        logger.debug("K-Means fertig. Cluster-Zentren: %s", centers)

    def predict(self, feature: np.ndarray) -> int:
        """Sagt das Team für einen einzelnen HSV-Feature-Vektor vorher.

        Args:
            feature: HSV-Vektor mit Shape (3,).

        Returns:
            ``TEAM_A`` (0) oder ``TEAM_B`` (1).

        Raises:
            RuntimeError: Falls ``fit()`` noch nicht aufgerufen wurde.
        """
        if not self._fitted or self._centers is None:
            raise RuntimeError("fit() muss vor predict() aufgerufen werden.")

        dists = np.linalg.norm(self._centers - feature.astype(np.float32), axis=1)
        return int(np.argmin(dists))

    def get_team(self, track_id: int, feature: np.ndarray) -> int:
        """Gibt das Team für einen Spieler zurück.

        Manuelle Korrekturen (``override``) haben Vorrang vor dem K-Means-Ergebnis.

        Args:
            track_id: Stabile ByteTrack-ID des Spielers.
            feature:  Aktueller HSV-Merkmals-Vektor.

        Returns:
            ``TEAM_A``, ``TEAM_B`` oder ``TEAM_UNKNOWN``.
        """
        if track_id in self._overrides:
            return self._overrides[track_id]
        if not self._fitted:
            return TEAM_UNKNOWN
        return self.predict(feature)

    # ------------------------------------------------------------------
    # Manuelle Korrektur (Should: bleibt über gesamten Clip erhalten)
    # ------------------------------------------------------------------

    def override(self, track_id: int, team: int) -> None:
        """Überschreibt die automatische Zuordnung für eine Track-ID.

        Die Korrektur gilt für alle zukünftigen und vergangenen Anfragen
        an ``get_team()`` für diese ``track_id``.

        Args:
            track_id: ByteTrack-ID des Spielers.
            team:     ``TEAM_A`` oder ``TEAM_B``.
        """
        if team not in (TEAM_A, TEAM_B):
            raise ValueError(f"team muss TEAM_A ({TEAM_A}) oder TEAM_B ({TEAM_B}) sein.")
        self._overrides[track_id] = team
        logger.debug("Manuelle Korrektur: track_id=%d → team=%d", track_id, team)

    def remove_override(self, track_id: int) -> None:
        """Entfernt eine manuelle Korrektur für eine Track-ID."""
        self._overrides.pop(track_id, None)

    @property
    def overrides(self) -> dict[int, int]:
        """Gibt eine Kopie aller manuellen Korrekturen zurück."""
        return dict(self._overrides)

    # ------------------------------------------------------------------
    # Persistenz (Could: historische Zuordnung vorladen)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Speichert Cluster-Zentren und manuelle Korrekturen als JSON.

        Args:
            path: Zieldatei (.json).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "overrides": {str(k): v for k, v in self._overrides.items()},
            "centers": self._centers.tolist() if self._centers is not None else None,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("TeamAssigner gespeichert: %s", path)

    def load(self, path: str | Path) -> None:
        """Lädt eine zuvor gespeicherte Zuordnung aus einer JSON-Datei.

        Args:
            path: Quelldatei (.json), muss existieren.

        Raises:
            FileNotFoundError: Falls die Datei nicht existiert.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {path}")

        data = json.loads(path.read_text())
        self._overrides = {int(k): int(v) for k, v in data.get("overrides", {}).items()}
        centers = data.get("centers")
        if centers is not None:
            self._centers = np.array(centers, dtype=np.float32)
            self._fitted = True
        logger.info("TeamAssigner geladen: %s", path)
