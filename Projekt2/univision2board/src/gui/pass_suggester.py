"""
Regelbasierte Passvorschläge auf dem Taktikboard.

Algorithmus:
  Für jeden Spieler von Team T werden Pässe zu Mitspielern vorgeschlagen,
  sofern:
  1. Der Mitspieler innerhalb von `max_pass_dist` Pixeln liegt.
  2. Kein Gegner innerhalb eines Korridors von `block_radius` Pixeln
     auf der direkten Passlinie steht (einfache Sicht-Linie).
"""

from typing import Dict, List, Tuple

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


class PassSuggestion:
    """Ein einzelner Passvorschlag zwischen zwei Spielern."""

    def __init__(
        self,
        from_id: int,
        to_id: int,
        from_pos: Tuple[float, float],
        to_pos: Tuple[float, float],
    ):
        self.from_id = from_id
        self.to_id = to_id
        self.from_pos = from_pos
        self.to_pos = to_pos


class PassSuggester:
    """
    Erzeugt regelbasierte Passvorschläge aus aktuellen Spielerpositionen.

    Parameters:
        max_pass_dist:  Maximale Distanz (px) für einen Pass.
        block_radius:   Korridor-Breite (px) – Gegner darin blockiert den Pass.
        max_per_player: Maximale Anzahl Passvorschläge pro Spieler.
    """

    def __init__(
        self,
        max_pass_dist: float = 350.0,
        block_radius: float = 30.0,
        max_per_player: int = 2,
    ):
        self.max_pass_dist = max_pass_dist
        self.block_radius = block_radius
        self.max_per_player = max_per_player

    def suggest(
        self,
        positions: Dict[int, Tuple[float, float]],
        teams: Dict[int, int],
    ) -> List[PassSuggestion]:
        """
        Berechnet Passvorschläge für alle Spieler.

        Parameters:
            positions: {track_id: (board_x, board_y)}
            teams:     {track_id: team_id (0=A, 1=B)}

        Returns:
            Liste von PassSuggestion-Objekten.
        """
        suggestions: List[PassSuggestion] = []
        seen_pairs = set()

        for from_id, from_pos in positions.items():
            from_team = teams.get(from_id)
            if from_team is None:
                continue

            # Gegner-Positionen für Blockade-Prüfung
            opponent_positions = [
                pos for tid, pos in positions.items()
                if teams.get(tid) is not None and teams[tid] != from_team
            ]

            # Mitspieler nach Distanz sortieren
            teammates = [
                (tid, pos)
                for tid, pos in positions.items()
                if teams.get(tid) == from_team and tid != from_id
            ]
            teammates.sort(key=lambda t: self._dist(from_pos, t[1]))

            count = 0
            for to_id, to_pos in teammates:
                if count >= self.max_per_player:
                    break
                pair = (min(from_id, to_id), max(from_id, to_id))
                if pair in seen_pairs:
                    continue
                if self._dist(from_pos, to_pos) > self.max_pass_dist:
                    continue
                if self._is_blocked(from_pos, to_pos, opponent_positions):
                    continue
                suggestions.append(PassSuggestion(from_id, to_id, from_pos, to_pos))
                seen_pairs.add(pair)
                count += 1

        return suggestions

    # ------------------------------------------------------------------
    # Hilfsfunktionen
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.hypot(b[0] - a[0], b[1] - a[1]))

    def _is_blocked(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        opponents: List[Tuple[float, float]],
    ) -> bool:
        """Gibt True zurück, wenn ein Gegner die Passlinie a→b blockiert."""
        ax, ay = a
        bx, by = b
        length = self._dist(a, b)
        if length < 1e-6:
            return False
        dx, dy = (bx - ax) / length, (by - ay) / length

        for ox, oy in opponents:
            # Projektion des Gegners auf die Passlinie
            t = (ox - ax) * dx + (oy - ay) * dy
            if t < 0 or t > length:
                continue
            # Senkrechter Abstand des Gegners von der Linie
            perp_dist = abs((ox - ax) * dy - (oy - ay) * dx)
            if perp_dist < self.block_radius:
                return True
        return False
