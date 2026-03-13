Schrittweiser Umsetzungsplan für UniVision2Board MVP
Phase 1: Projekt-Setup & Infrastruktur
Abschnitte
1. Python-Projektstruktur erstellen
2. Abhängigkeiten installieren
3. Grundlegende Test- und Logging-Struktur einrichten
Schritte
Abschnitt 1: Projektstruktur
•Neues Python-Projekt initialisieren (univision2board/)
•Verzeichnisse anlegen:
/univision2board
/src
video_processing/
object_detection/
tracking/
gui/
utils/
/tests
requirements.txt
README.md
•
Git-Repository initialisieren
Abschnitt 2: Abhängigkeiten
•
•
requirements.txt erstellen mit:
oopencv-python
otorch
otorchvision
oyolov5 / YOLOv11
onumpy
opandas
oscikit-learn
opyqt5 (oder tkinter)
omatplotlib (optional für Debugging)
Testen, dass alle Pakete importierbar sind
Abschnitt 3: Logging & Tests
•Logging konfigurieren (logging-Modul)
•Basis-Teststruktur aufsetzen (pytest)
•Erstes Dummy-Testskript, das einen Frame lädt und prüft, dass er korrekt gelesen
wirdPhase 2: Videoverarbeitung & Homography
Abschnitte
1. Video laden und Frame-Extraktion
2. Manuelle Kalibrierung der Spielfeldpunkte
3. Berechnung der Homography-Matrix
4. Transformation von Spielerkoordinaten auf Taktikboard
Schritte
Abschnitt 1: Video laden
•OpenCV VideoCapture implementieren
•Frames nacheinander lesen und in numpy-Array speichern
•Test: Prüfen, dass alle Frames korrekt extrahiert werden
Abschnitt 2: Manuelle Kalibrierung
•GUI-Dialog, um 6–8 Punkte auf dem Spielfeld zu klicken (Bullypunkte, Torzentrum)
•Koordinaten speichern
•Test: Punkte lassen sich korrekt auswählen und speichern
Abschnitt 3: Homography
•OpenCV findHomography mit ausgewählten Punkten implementieren
•Funktion schreiben, die Videokoordinaten → Taktikboard-Koordinaten transformiert
•Test: Transformation überprüfen (z. B. Spielerkreuz an bekannten Punkten)
Phase 3: Objekterkennung
Abschnitte
1. YOLOv11-Modell laden (vortrainiert)
2. Spieler, Torhüter und optional Ball erkennen
3. Bounding-Boxes und Klassenlabels zurückgeben
Schritte
Abschnitt 1: Modell laden
•YOLOv11 vortrainiertes Modell einbinden
•Test: Dummy-Image erkennen lassen und Bounding-Boxen prüfen
Abschnitt 2: Objekterkennung auf Video
•Frames durch YOLO schicken
•Bounding-Boxen, Klasse und Confidence speichern
•Test: Überprüfen, dass Spieler erkannt werden (manuell / automatisiert)Phase 4: Spieler-Tracking
Abschnitte
1. ByteTrack einbinden
2. IDs für Spieler über Frames konsistent halten
3. Laufwege für ausgewählte Spieler erstellen
Schritte
Abschnitt 1: Tracking implementieren
•Bounding-Boxes YOLO → ByteTrack übergeben
•IDs erzeugen und über Frames verfolgen
•Test: Spieler über mehrere Frames verfolgen und IDs korrekt zuordnen
Abschnitt 2: Laufwege extrahieren
•Mittelpunkt jeder Bounding-Box speichern
•Funktion, die Laufweg als Liste von Punkten zurückgibt
•Test: Punkte korrekt auf Taktikboard transformiert
Phase 5: Teamzuordnung
Abschnitte
1. Automatisches Color-Clustering
2. Trainer-Korrektur
3. Speicherung der Teamzuordnung
Schritte
Abschnitt 1: Color-Clustering
•HSV-Farbraum aus Bounding-Box extrahieren
•K-Means (k=2) durchführen → zwei Teams
•Test: Spieler korrekt gruppiert
Abschnitt 2: Trainer-Korrektur
•GUI-Interaktion: Spieler manuell verschieben oder Team wechseln
•Test: Korrektur wird gespeichert und auf restlichen Clip angewendet
Phase 6: Taktikboard GUI
Abschnitte
1. Grundlegende GUI erstellen (PyQt5 / Tkinter)
2. Spieler und Ball visualisieren (X, O, T, .)
3. Laufwege für ausgewählte Spieler anzeigen
4. Passvorschläge darstellen5. Interaktive Zeichenfunktionen einbauen
Schritte
Abschnitt 1: GUI-Basis
•Fenster mit Canvas erstellen
•Test: Fenster öffnet sich und kann Frames anzeigen
Abschnitt 2: Spieler visualisieren
•Symbole für Team / Torhüter / Ball einzeichnen
•Test: Symbole erscheinen an transformierten Positionen
Abschnitt 3: Laufwege
•Funktion, die für ausgewählten Spieler den Laufweg zeichnet
•Test: Linie korrekt gezeichnet, auf Klick auswählbar
Abschnitt 4: Passvorschläge
•Regelbasierte Logik implementieren
•Pfeile zu möglichen Passempfängern einzeichnen
•Test: Pfeile korrekt generiert
Abschnitt 5: Manuelles Zeichnen
•Trainer kann Pässe, Schüsse, Laufwege zeichnen
•Test: Zeichnung bleibt gespeichert / exportierbar
Phase 7: Datenexport
Abschnitte
1. Screenshot / Bild exportieren (PNG/PDF)
2. Video mit Overlay exportieren
3. Temporäre Anzeige ohne Speicherung
Schritte
Abschnitt 1: Bildexport
•Canvas als PNG/PDF speichern
•Test: Datei korrekt erstellt
Abschnitt 2: Videoexport
•Frames mit Overlay speichern → cv2.VideoWriter
•Test: Video korrekt mit Taktikboard exportiert
Abschnitt 3: Temporäre Anzeige
•Option, alles nur temporär zu zeigen
•Test: Board zeigt korrekt an, keine Daten persistentPhase 8: Integration & Test
Abschnitte
1. Alle Module verbinden (Video → YOLO → Tracking → Team → GUI)
2. Workflow testen
3. Fehlerbehandlung implementieren
Schritte
Abschnitt 1: Module verbinden
•Datenfluss definieren: Video → Frame → Objekterkennung → Tracking →
Transformation → GUI → Export
•Test: End-to-End-Durchlauf mit Dummy-Video
Abschnitt 2: Workflow-Test
•Trainer lädt Clip → Spieler erkennen → Laufwege → Passvorschläge → Export
•Test: Alles funktioniert ohne Unterbrechung
Abschnitt 3: Fehlerhandling
•Fehlende Ballerkennung, falsche Teamzuordnung, Tracking-Ausfälle abfangen
•Test: GUI zeigt Warnungen / Korrekturmöglichkeiten
Damit haben wir einen vollständigen, schrittweisen MVP-Plan, der:
•iterativ aufgebaut ist
•aufeinander aufbauende kleine Schritte enthält
•Testbarkeit in jedem Schritt sicherstellt
•Integration der Module am Ende gewährleistet