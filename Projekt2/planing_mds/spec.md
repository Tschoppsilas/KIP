# UniVision2Board – Technische Spezifikation (MVP, Fokus Spieler)

## Ziel
Ein Unihockey-Video soll automatisiert auf ein digitales Taktikboard übertragen werden, damit Trainer Spielszenen schneller analysieren und gemeinsam mit dem Team besprechen koennen.

**Eingrenzung dieses MVP:** Die Pipeline konzentriert sich auf die **Erkennung und Verfolgung von Spielern (Feldspielern)**. Es gibt **keine** gesonderte Erkennung oder Darstellung von Torhueter, Schiedsrichter oder weiteren Rollen im MVP-Scope.

## Programmiersprache
- **Python** als Hauptsprache, wegen starker Bibliotheken fuer Computer Vision, Tracking, KI, Videoverarbeitung und GUI.

## Input
- Video-Datei eines Unihockeyspiels (Post-Game-Analyse, kein Echtzeitbetrieb).
- Manuelle Kalibrierpunkte fuer die Spielfeld-Transformation (Homography).

## Verarbeitungs-Pipeline

### 1) Videoverarbeitung
- **OpenCV** fuer Laden, Frame-Extraktion, Bearbeitung und Overlay-Ausgabe.
- Homography-Transformation von Videokoordinaten auf Taktikboard-Koordinaten.

### 2) Objekterkennung (nur Spieler)
- **YOLOv11 (vortrainiert oder feingetuned)** liefert Bounding-Boxes fuer **Spieler**.
- Ausgabe als Bounding-Boxes plus Metadaten (Confidence, ggf. eine Spieler-Klasse).
- Keine Pflicht im MVP: Torhueter, Schiedsrichter, Ball als eigene Ziele.

### 3) Spieler-Tracking
- **ByteTrack** zur stabilen Verfolgung von Spieler-Bounding-Boxes ueber Frames.
- Liefert konsistente IDs und bildet die Basis fuer Laufwege.

### 4) Teamzuordnung
- Automatische Zuordnung per **Color-Clustering** (z. B. K-Means im HSV-Farbraum) auf den erkannten Spieler-Regionen.
- Manuelle **Trainer-Korrektur** in der GUI bei Fehlzuordnungen.

### 5) Taktik- und Passlogik
- Regelbasierte Passvorschlaege auf Basis sichtbarer Mitspieler und blockierter Passlinien.
- Interaktive Zeichnung von taktischen Elementen (Paesse, Schuesse, Laufwege).

## Output

### Taktikboard-Visualisierung
- Desktop-GUI mit **PyQt5 oder Tkinter**.
- Darstellung der **Spieler** (z. B. pro Team), Laufwegen und Passvorschlaegen — ohne Pflichtsymbole fuer Torhueter oder Schiris im MVP.

### Export
- Bildexport als **PNG/PDF**.
- Videoexport mit Overlay via **OpenCV VideoWriter**.

## MVP-Umfang
- **Spieler** erkennen und tracken.
- Teamzuordnung automatisch mit manueller Korrektur.
- Laufwege visualisieren.
- Regelbasierte Passvorschlaege anzeigen.
- Trainer-Interaktion auf dem Taktikboard ermoeglichen.
- Szenen als Bild oder Video exportieren.
- **Stable-Baselines3 optional im MVP** fuer spaetere, weitergehende Spielzuglogik.

## Stretch Goals (nicht MVP)
- Dedizierte Erkennung von Torhueter, Ball oder Schiedsrichter.
- Erweiterte taktische Vorschlaege (komplexere Spielzuege).
- Automatische Erkennung spezifischer Spielsituationen.
- Erweiterte Statistik und Analysefunktionen.

## Grenzen und Rahmenbedingungen
- Das System ist als **unterstuetzendes Analysewerkzeug** gedacht, nicht als autonomes Entscheidungssystem.
- Kein sicherheitskritischer Einsatz vorgesehen.
- Videomaterial kann personenbezogene Daten enthalten; Datenschutz ist bei Weiterentwicklung zu beachten.
