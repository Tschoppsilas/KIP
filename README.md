# UniVision2Board (MVP)

UniVision2Board ist ein Python-Projekt zur Post-Game-Analyse von Unihockey-Videos.
Der Fokus liegt auf der Uebertragung von Spielszenen auf ein digitales Taktikboard.

## Projekt-Scope (MVP)
- Post-Game-Analyse von Videodateien
- Kein Echtzeitbetrieb

## Projektstruktur (Phase 1)
- `src/video_processing`: Videoeinlesen und Frame-Verarbeitung
- `src/object_detection`: Platzhalter fuer Erkennungskomponenten
- `src/tracking`: Platzhalter fuer Tracking-Komponenten
- `src/gui`: Platzhalter fuer GUI-Komponenten
- `src/utils`: Logging und Hilfsfunktionen
- `tests`: Pytest-Testfaelle

## Schnellstart
1. Virtuelle Umgebung erstellen und aktivieren:
   - Linux/macOS: `python -m venv .venv && source .venv/bin/activate`
2. Abhaengigkeiten installieren:
   - `pip install -r requirements.txt`
   - Optional fuer Debug-Visualisierung (Phase 1 Could): `pip install -r requirements-debug.txt`
   - Fuer spaetere Erkennungsphase (Phase 3): `pip install -r requirements-phase3.txt`
3. Tests ausfuehren:
   - `pytest`

## Ausfuehren (End-to-End Pipeline)
Das Hauptskript ist `run_full_pipeline.py`.

- Beispiel:
  - `python3 run_full_pipeline.py Videos/Trainingsdaten/Mittelland_4.mp4`

Beim Start erscheint ein Setup-Dialog:
- Kalibrierung im Video (6 Punkte)
- Vorschau/Pruefung auf dem Taktikboard
- Spieler sampeln (Team A/B + Schiri)

## Ordner
- `outputs/`: generierte Videos/Logs (nicht versioniert)
- `scripts/`: Hilfsskripte (Kalibrierung, Visualisierung, Demos)
- `docs/`: Notizen/TODO
- `archive/`: alte/abgelegte Dateien (z. B. Kalibrierungs-JSONs)