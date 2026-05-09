# UniVision2Board MVP Checkliste (Fokus: Spieler-Erkennung)

> MVP-Scope: **Spieler** erkennen, tracken, zuordnen — **keine** Pflicht fuer Torhueter- oder Schiri-Erkennung/-Darstellung. Archivierte Vollversion: `Archiv/`.

## Phase 1: Projekt-Setup und Infrastruktur
### Must
- [ ] Python-Projekt initialisiert (`univision2board/`).
- [ ] Verzeichnisse erstellt (`src/video_processing`, `src/object_detection`, `src/tracking`, `src/gui`, `src/utils`, `tests`).
- [ ] `requirements.txt` mit Kernabhaengigkeiten erstellt und installierbar.
- [ ] Logging-Basis konfiguriert.
- [ ] `pytest`-Basisstruktur eingerichtet.
- [ ] Scope dokumentiert: Post-Game-Analyse, kein Echtzeitbetrieb; **Fokus Spieler-Erkennung**.

### Should
- [ ] Dummy-Test implementiert: Ein Frame wird aus Video geladen und validiert.
- [ ] README mit Startanleitung angelegt.

### Could
- [ ] Optionales Debugging-Setup (z. B. Matplotlib) vorbereiten.

### Definition of Done
- [ ] Umgebung laeuft reproduzierbar.
- [ ] Testlauf (`pytest`) ohne Fehler.

## Phase 2: Videoverarbeitung und Homography
### Must
- [ ] Video mit OpenCV laden und Frames extrahieren.
- [ ] GUI-Dialog fuer manuelle Kalibrierung (6-8 Punkte) umgesetzt.
- [ ] Kalibrierungspunkte werden gespeichert.
- [ ] Homography-Matrix berechnet und angewendet.

### Should
- [ ] Kalibrierung auf mehreren Szenen kurz gegentesten.

### Could
- [ ] Vorlagenpunkte fuer wiederkehrende Kamerawinkel speichern.

### Definition of Done
- [ ] Videokoordinaten werden plausibel auf Board-Koordinaten transformiert.

## Phase 3: Objekterkennung (nur Spieler)
### Must
- [ ] YOLOv11 (vortrainiert oder feingetuned) eingebunden.
- [ ] Pipeline liefert pro Frame Bounding-Boxes fuer **Spieler**; keine MVP-Pflicht fuer Torhueter oder Schiedsrichter.
- [ ] Bounding-Boxes, Confidence (und ggf. Klasse) werden sauber ausgegeben.
- [ ] Dokumentiert: wie Nicht-Spieler (Zuschauer, Bank) reduziert werden (ROI, Training, Filter).

### Should
- [ ] Dummy-Frame-Test fuer Spieler-Detection-Ausgabe hinterlegt.

### Could
- [ ] Confidence-Schwellen konfigurierbar machen.

### Definition of Done
- [ ] Detection-Output ist stabil und als Input fuer Tracking nutzbar (**Spieler-Fokus**).

## Phase 4: Spieler-Tracking
### Must
- [ ] ByteTrack integriert.
- [ ] Spieler-IDs bleiben ueber Frames konsistent.
- [ ] Laufwege aus Mittelpunkten pro Bounding-Box erzeugt.

### Should
- [ ] Trajektorien visuell auf dem Taktikboard kontrollierbar.

### Could
- [ ] Glattung der Laufwege fuer bessere Lesbarkeit ergaenzen.

### Definition of Done
- [ ] Laufwege fuer ausgewaehlte Spieler sind korrekt transformiert und darstellbar.

## Phase 5: Teamzuordnung
### Must
- [ ] HSV-Farbmerkmale aus **Spieler**-Bounding-Boxes extrahieren.
- [ ] K-Means (`k=2`) fuer Team A/B implementieren.
- [ ] Teamlabels im UI sichtbar machen.
- [ ] Trainer-Korrektur fuer Fehlzuordnung ermoeglichen.

### Should
- [ ] Korrekturen fuer den restlichen Clip beibehalten.

### Could
- [ ] Historische Teamzuordnung fuer spaetere Clips vorladen.

### Definition of Done
- [ ] Teamzuordnung funktioniert automatisch und ist manuell korrigierbar.

## Phase 6: Taktikboard GUI
### Must
- [ ] GUI-Fenster mit Board/Canvas erstellt (PyQt5 oder Tkinter).
- [ ] Symbole/Darstellung fuer **Team A / Team B** (Spieler); **keine** Pflichtsymbole fuer Torhueter oder Schiri im MVP.
- [ ] Laufwege fuer ausgewaehlte Spieler visualisiert.
- [ ] Regelbasierte Passvorschlaege als Pfeile dargestellt.
- [ ] Interaktive Zeichenfunktionen fuer Pass, Schuss, Laufweg vorhanden.

### Should
- [ ] Trainer kann Spieler/Laufwege auswaehlen und korrigieren.

### Could
- [ ] Touch-optimierte Interaktion fuer Tablet verbessern.
- [ ] Ball oder Torhueter spaeter als Stretch visualisieren.

### Definition of Done
- [ ] Board ist bedienbar und taktische Elemente bleiben sichtbar/weiter nutzbar.

## Phase 7: Datenexport
### Must
- [ ] Bildexport als PNG/PDF funktioniert.
- [ ] Videoexport mit Overlay via `cv2.VideoWriter` funktioniert.

### Should
- [ ] Temporaere Anzeige ohne persistente Speicherung moeglich.

### Could
- [ ] Export-Presets (Qualitaet/Format) ergaenzen.

### Definition of Done
- [ ] Exportierte Dateien sind oeffnbar und enthalten alle sichtbaren Overlays.

## Phase 8: Integration und Tests
### Must
- [ ] End-to-End-Datenfluss verbunden (`Video -> Spieler-Detection -> Tracking -> Team -> GUI -> Export`).
- [ ] Workflow-Test aus Trainersicht durchgefuehrt.
- [ ] Fehlerfaelle abgefangen (falsche Teamzuordnung, Tracking-Ausfall, Ausreisser in der Spieler-Detection).

### Should
- [ ] GUI zeigt klare Warnungen und Korrekturmoeglichkeiten.
- [ ] Datenschutz-Hinweis fuer potenziell personenbezogene Videodaten im Projektablauf dokumentieren.

### Could
- [ ] Kurzes Testprotokoll pro Demo-Clip dokumentieren.

### Definition of Done
- [ ] End-to-End-Integration laeuft stabil ohne manuelle Codeeingriffe waehrend des Ablaufs.

## Optional / Stretch Goals
### Should
- [ ] Ballerkennung verbessern (klein/unscharf).
- [ ] Automatische Szenenerkennung integrieren.
- [ ] Statistische Auswertungen (Paesse, Schuesse, Positionen) ergaenzen.
- [ ] Torhueter oder Schiri als eigene Klasse/Darstellung.

### Could
- [ ] Pose Estimation fuer Spielerorientierung integrieren.
- [ ] Komplexere KI-basierte Spielzuege vorschlagen.
- [ ] Stable-Baselines3 als optionales Modul fuer lernbasierte Spielzuglogik evaluieren.
