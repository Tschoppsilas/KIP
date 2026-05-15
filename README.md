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
| Älterer Stand (Auf Grund von Ordner verschiebungen um Platz zu sparen, ist das Programm nicht mehr Funktionsfähig Ich behalte es trotzdem drin, da es für gewisse Informationen noch wichtig sein könnte) | [Projekt1/README.md](Projekt1/README.md) |

Für **Projekt 2** sind im Repository vorgesehen (siehe Ausnahmen in `.gitignore`): Taktikboard-PNG, das trainierte Gewicht `best.pt` (über **Git LFS**) und das Demo-Video **`Projekt2/Videos/Abgabe_Demo.mp4`** (sofern mit committed und gepusht). Alles andere (z. B. volle Trainingsruns, Rohdaten) bleibt lokal.

## Setup (Git LFS)

Große Binärdateien (z. B. `*.pt`) werden mit **Git LFS** versioniert ([git-lfs.com](https://git-lfs.com)).

```bash
git lfs install
git clone <repo-url>
cd <repo>
git lfs pull
```

**Nach dem Klonen prüfen:** `Projekt2/finetune/runs/train/weights/best.pt` sollte **ca. 100 MB** groß sein (echtes Gewicht), nicht nur ein kleiner Textpointer.

**Video:** Entweder `Projekt2/Videos/Abgabe_Demo.mp4` verwenden (falls im Repo) oder ein beliebiges MP4 nach `Projekt2/Videos/` legen und im README der App den Dateinamen anpassen.
