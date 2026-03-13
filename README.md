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