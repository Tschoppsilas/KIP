"""
PlayerActionDialog – kompakter Dialog zur Spieler-Verwaltung im laufenden Betrieb.

Wird geöffnet wenn der Trainer im TacticBoardWidget auf einen Spieler klickt.
Mögliche Aktionen: Team wechseln (A / B) oder Spieler dauerhaft löschen.
"""

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class PlayerActionDialog(QDialog):
    """
    Kleiner modaler Dialog: »Spieler #N — Team: A/B«

    Nach ``exec_()`` liefert ``get_action()`` einen der Werte:
        'team_a'  – Spieler Team A zuordnen
        'team_b'  – Spieler Team B zuordnen
        'delete'  – Spieler für den Rest des Videos ausblenden
        None      – Abgebrochen
    """

    def __init__(self, track_id: int, current_team: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spieler-Aktion")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)
        self.setMinimumWidth(320)
        self._action: Optional[str] = None

        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 12, 16, 12)

        team_label = "A" if current_team == 0 else "B"
        info = QLabel(f"<b>Spieler #{track_id}</b> &nbsp;·&nbsp; aktuell <b>Team {team_label}</b>")
        info.setAlignment(Qt.AlignCenter)
        root.addWidget(info)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        btn_a = QPushButton("⬡  Team A")
        btn_a.setStyleSheet("font-weight: bold; color: #d23232;")
        btn_a.clicked.connect(lambda: self._finish("team_a"))
        btn_row.addWidget(btn_a)

        btn_b = QPushButton("⬡  Team B")
        btn_b.setStyleSheet("font-weight: bold; color: #3264dc;")
        btn_b.clicked.connect(lambda: self._finish("team_b"))
        btn_row.addWidget(btn_b)

        btn_del = QPushButton("🗑  Löschen")
        btn_del.setStyleSheet("font-weight: bold; color: #c04000;")
        btn_del.setToolTip("Spieler für den Rest des Videos ausblenden")
        btn_del.clicked.connect(lambda: self._finish("delete"))
        btn_row.addWidget(btn_del)

        btn_cancel = QPushButton("Abbrechen")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        root.addLayout(btn_row)

    def _finish(self, action: str) -> None:
        self._action = action
        self.accept()

    def get_action(self) -> Optional[str]:
        """Gibt die gewählte Aktion zurück (nur nach ``accept()``)."""
        return self._action
