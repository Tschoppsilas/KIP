# Schrittweiser Umsetzungsplan fuer UniVision2Board (MVP, Fokus Spieler)

## Phase 1: Projekt-Setup und Infrastruktur
### Schritte
- Python-Projektstruktur anlegen (`src`, `tests`, Module fuer Verarbeitung, Erkennung, Tracking, GUI, Utils).
- `requirements.txt` aufsetzen und Kernpakete installierbar machen.
- Logging und Basis-Tests mit `pytest` einrichten.
- Projekt-Scope festhalten: Post-Game-Analyse, kein Echtzeitbetrieb; **MVP nur Spieler-Erkennung/-Tracking**, keine Pflicht fuer Torhueter/Schiris.

### Deliverable
- Laufendes Projektgeruest mit reproduzierbarer Umgebung und erstem Testlauf.

### Done-Kriterium
- `pytest` laeuft erfolgreich.
- Ein Dummy-Test liest einen Video-Frame korrekt ein.

### Abhaengigkeiten und Risiken
- **Abhaengigkeit:** Keine (Startphase).
- **Risiko:** Paketkonflikte (Torch/YOLO/OpenCV) bremsen den Start.

## Phase 2: Videoverarbeitung und Homography
### Schritte
- Video mit OpenCV laden und Frames verarbeiten.
- GUI-gestuetzte Kalibrierung fuer 6-8 Spielfeldpunkte umsetzen.
- Homography berechnen und Punkte auf das Taktikboard transformieren.

### Deliverable
- Verlaessliche Koordinatentransformation Video -> Taktikboard.

### Done-Kriterium
- Kalibrierungspunkte koennen gespeichert und wiederverwendet werden.
- Testpunkte landen nachvollziehbar an den erwarteten Board-Positionen.

### Abhaengigkeiten und Risiken
- **Abhaengigkeit:** Phase 1.
- **Risiko:** Ungenaue manuelle Klicks fuehren zu verzerrter Transformation.

## Phase 3: Objekterkennung (nur Spieler)
### Schritte
- Vortrainiertes oder feingetuntes YOLOv11-Modell integrieren.
- **Nur Spieler** pro Frame als Zielsetzung: eine Spieler-Klasse oder Filterung/Training so, dass Bounding-Boxes **Feldspieler** abbilden — **kein** MVP-Zwang fuer Torhueter oder Schiedsrichter.
- Bounding-Boxes, Klassen (falls vorhanden) und Confidence-Werte speichern.

### Deliverable
- Stabiler Detection-Output **fuer Spieler** pro Frame als Eingabe fuer Tracking.

### Done-Kriterium
- Auf Testsequenzen werden **Spieler** verlaesslich erkannt.
- Detection-Daten sind im einheitlichen Format verfuegbar.

### Abhaengigkeiten und Risiken
- **Abhaengigkeit:** Phase 1 (Umgebung), Phase 2 (Frame-Pipeline).
- **Risiko:** Verdeckungen/Bewegungsunschaerfe senken Erkennungsqualitaet.
- **Risiko:** Generische Personendetektion erfasst Zuschauer/Bank — ggf. ROI oder Training noetig.

## Phase 4: Spieler-Tracking
### Schritte
- ByteTrack an den YOLO-Output anbinden.
- IDs ueber Frames konsistent halten.
- Laufwege als Punktfolgen pro Spieler erzeugen.

### Deliverable
- Nachvollziehbare Spielertrajektorien mit stabilen IDs.

### Done-Kriterium
- Spieler bleiben ueber laengere Sequenzen derselben ID zugeordnet.
- Laufwege koennen auf das Taktikboard uebertragen werden.

### Abhaengigkeiten und Risiken
- **Abhaengigkeit:** Phase 3.
- **Risiko:** ID-Switches bei Ueberdeckung oder schnellem Richtungswechsel.

## Phase 5: Teamzuordnung
### Schritte
- Trikotfarben (HSV) aus **Spieler**-Bounding-Boxes extrahieren und per K-Means (`k=2`) clustern.
- Teamlabels in der GUI anzeigen.
- Manuelle Korrekturfunktion fuer Trainer bereitstellen.

### Deliverable
- Automatische Teamzuordnung mit korrigierbarer Benutzeroberflaeche.

### Done-Kriterium
- Team A/B wird in typischen Szenen sinnvoll getrennt.
- Korrekturen werden gespeichert und im Clip beibehalten.

### Abhaengigkeiten und Risiken
- **Abhaengigkeit:** Phase 3 und 4.
- **Risiko:** Aehnliche Trikotfarben oder Lichtwechsel verfalschen Clustering.

## Phase 6: Taktikboard-GUI
### Schritte
- GUI-Basis (PyQt5 oder Tkinter) mit Zeichenflaeche bauen.
- **Spieler** (z. B. Team A/B), Laufwege und Passpfeile visualisieren — ohne verpflichtende Darstellung von Torhueter oder Schiri im MVP.
- Interaktive Zeichenfunktionen fuer Pass, Schuss und Laufweg einbauen.

### Deliverable
- Bedienbares Taktikboard fuer Analyse und Trainerinteraktion.

### Done-Kriterium
- Spieler erscheinen an transformierten Positionen.
- Trainer kann markieren, korrigieren und taktische Elemente einzeichnen.

### Abhaengigkeiten und Risiken
- **Abhaengigkeit:** Phase 2, 4, 5.
- **Risiko:** GUI-Performance sinkt bei langen Sequenzen.

## Phase 7: Datenexport
### Schritte
- Export als PNG/PDF implementieren.
- Videoexport mit Overlay via `cv2.VideoWriter` ermoeglichen.
- Option fuer rein temporaere Anzeige ohne persistente Speicherung anbieten.

### Deliverable
- Vollstaendige Ausgabefunktionen fuer Bild und Video.

### Done-Kriterium
- Exportdateien sind oeffnbar und enthalten alle sichtbaren Overlays.
- Temporaere Anzeige erzeugt keine ungewollten Dateien.

### Abhaengigkeiten und Risiken
- **Abhaengigkeit:** Phase 6.
- **Risiko:** Codec-/Formatprobleme bei Videoexport auf verschiedenen Systemen.

## Phase 8: Integration und Gesamt-Test
### Schritte
- End-to-End-Datenfluss verbinden: Video -> Spieler-Erkennung -> Tracking -> Team -> GUI -> Export.
- Workflow aus Trainersicht durchspielen.
- Fehlerfaelle abfangen (falsche Teamzuordnung, Tracking-Ausfall, zu wenige/zu viele Spieler-Detektionen).

### Deliverable
- Integrierter MVP-Prototyp mit durchgaengigem Workflow.

### Done-Kriterium
- Kompletter Ablauf funktioniert ohne manuelle Eingriffe im Code.
- Relevante Fehlerfaelle zeigen verstaendliche GUI-Hinweise.

### Abhaengigkeiten und Risiken
- **Abhaengigkeit:** Phase 1-7.
- **Risiko:** Schnittstellen zwischen Modulen sind inkonsistent.
- **Risiko:** Verarbeitung personenbezogener Videodaten erfordert sauberen Umgang mit Datenschutz bei Speicherung/Weitergabe.

## Optional im MVP / spaeter ausbauen
- Stable-Baselines3 als optionaler Baustein fuer weitergehende, lernbasierte Spielzuglogik.
- Torhueter, Ball, Schiedsrichter als zusaetzliche Klassen oder Symbole.
