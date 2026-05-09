# Workflow-Test — Trainersicht

## Ziel
End-to-End-Validierung: Kann ein Trainer ohne Programmierkenntnisse
die Pipeline starten und ein annotiertes Taktikboard-Video erzeugen?

---

## Testprotokoll (Demo-Clip: Muenchenstein_1.mp4)

### Voraussetzungen
- [x] Python 3.10+ installiert
- [x] `.venv` aktiviert (`source .venv/bin/activate`)
- [x] `requirements.txt` installiert
- [x] `best.pt` (trainiertes Modell) in `finetune/runs/train/weights/`
- [x] `calibration_muenchenstein1.json` vorhanden
- [x] `Taktikboard/Taktikboard.png` vorhanden

### Testschritte

| # | Aktion | Erwartetes Ergebnis | Status |
|---|--------|---------------------|--------|
| 1 | `python run_pipeline.py Videos/Muenchenstein_1.mp4` | Browser öffnet sich auf Port 5556 | ✓ |
| 2 | Klick auf Team-A-Spieler → Taste A | Box rot markiert, "TEAM A ✓" sichtbar | ✓ |
| 3 | Klick auf Team-B-Spieler → Taste B | Box blau markiert, "TEAM B ✓" sichtbar | ✓ |
| 4 | Klick auf "Speichern & Beenden" | `team_colors.json` erstellt, Browser meldet Erfolg | ✓ |
| 5 | Pipeline läuft automatisch weiter | Log zeigt K-Means übersprungen, Rendering startet | ✓ |
| 6 | Video wird nach Abschluss geöffnet | `output_annotated.mp4` öffnet sich im Player | ✓ |
| 7 | Split-View sichtbar (Video links, Board rechts) | Spielerpunkte auf Board, Team-Farben stabil | ✓ |

### Bekannte Einschränkungen

| Problem | Ursache | Workaround |
|---------|---------|------------|
| Langsame Verarbeitung (~3s/Frame) | Kein GPU, CPU-Only | Google Colab für Training verwenden |
| Spieler im Hintergrund manchmal nicht erkannt | Modell-Limitierung | Weitere Trainingsdaten annotieren |
| Team-Zuweisung bei ähnlichen Trikotfarben instabil | K-Means-Clustering | `pick_teams.py` für manuelle Kalibrierung verwenden |

### Performance-Messungen

| Clip | Frames | Dauer | Frames/s | Modell |
|------|--------|-------|----------|--------|
| Muenchenstein_1 | 90 | ~6 min | ~0.25 | best.pt (CPU) |
| Muenchenstein_1 | 300 | ~20 min | ~0.25 | best.pt (CPU) |

---

## Ergebnis-Bewertung

**Definition of Done (Phase 8):** ✓ End-to-End-Integration läuft stabil
ohne manuelle Codeeingriffe — ein einziger Befehl (`python run_pipeline.py`)
startet die gesamte Pipeline inkl. interaktiver Team-Kalibrierung.

**Offene Punkte:**
- [ ] Touch-optimierte Interaktion für Tablet (Phase 6 Could)
- [ ] Automatische Szenenerkennung (Stretch Goal)
- [ ] Statistik-Auswertungen (Pässe, Schüsse — Stretch Goal)
