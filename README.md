# KIP — UniVision2Board

Übergeordnetes Repository für die Arbeit am Projekt **UniVision2Board**: Videomaterial aus einem Unihockeyspiel soll für die **Video-Analyse** direkt auf ein digitales Taktikboard übertragen und dort weiterbearbeitet werden können.

## Zwei Projektstände

### Projekt 1 (`Projekt1/`)

Erste Umsetzung mit Pipeline, Training und vielen Komponenten in einem Verbund. In der Praxis hat sich gezeigt, dass Struktur und Komplexität schwer wartbar wurden und das Ergebnis **nicht den eigenen Ansprüchen** entsprach.

### Projekt 2 (`Projekt2/`)

Daraufhin wurde eine **vereinfachte, klarere Neuaufstellung** angelegt: Fokus auf eine schlanke Desktop-Pipeline (Paket `univision2board/`), klarer Scope und dokumentierte Bedienung. **Dies ist der empfohlene Einstieg.**

## Wo du weiterliest

| Bereich | Pfad |
|--------|------|
| Aktuelle App & Schnellstart | [Projekt2/univision2board/README.md](Projekt2/univision2board/README.md) |
| Scope (was MVP ist / was nicht) | [Projekt2/univision2board/SCOPE.md](Projekt2/univision2board/SCOPE.md) |
| Älterer Stand (Referenz) | [Projekt1/README.md](Projekt1/README.md) |

Videos, Taktikboard-Grafiken und Modellgewichte werden je nach Setup **lokal** oder über eigene Ablage verwaltet (siehe `.gitignore` und die jeweiligen READMEs).
