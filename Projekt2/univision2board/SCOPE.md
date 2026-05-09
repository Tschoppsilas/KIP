# UniVision2Board – Projektscope (MVP)

## Betriebsmodus
- **Post-Game-Analyse** (kein Echtzeitbetrieb)
- Lokale Desktop-Anwendung (Python)

## MVP-Fokus
- **Spieler-Erkennung und -Tracking** (Feldspieler)
- Keine Pflicht zur gesonderten Erkennung von Torwart, Schiedsrichter oder Ball

## Nicht im MVP
- Echtzeitverarbeitung
- Torwart-/Schiedsrichter-Erkennung als eigene Klasse
- Automatische Szenenerkennung
- Statistik-Auswertungen (Pässe, Schüsse)
- Stable-Baselines3 / lernbasierte Spielzuglogik

## Datenbasis
- Vortrainiertes YOLOv11-Modell: `finetune/runs/train/weights/best.pt`
- Testvideos: `Videos/` (Muenchenstein, Aarau, Mittelland)
- Taktikboard-Hintergrundbild: `Taktikboard/Taktikboard.png`
