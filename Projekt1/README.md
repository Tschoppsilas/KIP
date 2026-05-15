# UniVision2Board (Alt und funktioniert nicht mehr aufgrund von Verschiebungen von Ordner in Projekt2)

Automatische Post-Game-Analyse von Unihockey-Videos: Spieler werden erkannt,
getrackt und live auf ein digitales Taktikboard übertragen.

---

## Schnellstart

```bash
# 1. Umgebung einrichten (einmalig)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Pipeline starten
python run_pipeline.py Videos/Muenchenstein_1.mp4
```

Beim ersten Lauf öffnet sich automatisch ein **Browser-Fenster** zur
Team-Farb-Kalibrierung. Danach läuft alles vollständig automatisch.

---

## Vollständige Bedienung

```bash
python run_pipeline.py [VIDEO] [N_FRAMES] [OUTPUT] [CONF]

# Beispiele:
python run_pipeline.py Videos/Muenchenstein_1.mp4           # 300 Frames, Standard
python run_pipeline.py Videos/Muenchenstein_1.mp4 600       # 600 Frames
python run_pipeline.py Videos/Muenchenstein_1.mp4 300 out.mp4 0.18

# Team-Farb-Kalibrierung zurücksetzen:
python run_pipeline.py Videos/Muenchenstein_1.mp4 --reset-teams
```

### Team-Farb-Picker (einmalig pro Spiel)

```bash
python scripts/pick_teams.py Videos/Muenchenstein_1.mp4
```

Öffnet den Browser auf `http://localhost:5556`:
1. Klicke auf einen Spieler von Team A → Taste `A`
2. Klicke auf einen Spieler von Team B → Taste `B`
3. **Speichern** klicken → `team_colors.json` wird gespeichert

### Nur Pipeline (ohne Farb-Kalibrierung)

```bash
source .venv/bin/activate
python scripts/visualize_pipeline.py Videos/Muenchenstein_1.mp4 300 output.mp4 0.20
```

---

## Ausgabe

Das Ausgabevideo (`output_annotated.mp4`) zeigt:
- **Links**: Kamerabild mit Bounding-Boxes (Rot = Team A, Blau = Team B)
- **Rechts**: Taktikboard mit Live-Spielerpositionen

---

## Projektstruktur

```
run_pipeline.py              ← Haupt-Einstiegspunkt
scripts/
  visualize_pipeline.py      ← Video + Taktikboard rendern
  pick_teams.py              ← Interaktive Team-Farb-Kalibrierung
src/
  object_detection/          ← YOLO-Detektor
  tracking/                  ← ByteTrack + TeamAssigner
  video_processing/          ← Kalibrierung, Homographie
  gui/                       ← BoardRenderer, Taktikboard
finetune/
  dataset/                   ← Annotierte Trainingsdaten
  runs/train/weights/        ← Trainiertes Modell (best.pt)
  annotate.py                ← Browser-basiertes Annotationswerkzeug
  train.py                   ← Modell-Training (lokal)
  colab_training.ipynb       ← Training auf Google Colab (GPU)
Taktikboard/
  Taktikboard.png            ← Hintergrundbild für das digitale Board
```

---

## Kalibrierung (einmalig pro Kameraposition)

Falls keine `calibration_*.json` vorhanden ist, bitte mit dem
Kalibrierungstool neue Punkte setzen:

```bash
python -c "
from src.video_processing.calibration import CalibrationTool
CalibrationTool('Videos/Muenchenstein_1.mp4').run()
"
```

---

## Datenschutzhinweis

> **Personenbezogene Videodaten:** Die verarbeiteten Spielvideos können
> Personen identifizierbar zeigen. Es gelten die jeweiligen nationalen
> Datenschutzgesetze (CH: DSG, EU: DSGVO).
>
> - Videos sollten nur mit Einwilligung der abgebildeten Personen verarbeitet werden.
> - Ausgabedateien (`output_annotated.mp4`, Frames) nicht öffentlich teilen ohne Einwilligung.
> - Annotierte Trainingsdaten (`finetune/dataset/`) enthalten Spielerpositionen — ebenfalls vertraulich behandeln.
> - Das System speichert **keine** Videodaten remote; alle Verarbeitung erfolgt lokal.

---

## Abhängigkeiten

| Paket | Zweck |
|-------|-------|
| `ultralytics` | YOLO-Spielererkennung |
| `supervision` | ByteTrack-Tracking |
| `opencv-python` | Bildverarbeitung, VideoWriter |
| `Pillow` | Taktikboard-Rendering |
| `flask` | Browser-basierte Tools (Annotation, Team-Picker) |
| `scikit-learn` | K-Means Team-Clustering (Fallback) |
| `numpy` | Matrizenoperationen |
