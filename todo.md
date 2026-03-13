UniVision2Board MVP Checkliste
Phase 1: Projekt-Setup & Infrastruktur
•Python-Projekt initialisiert (univision2board/)
•Verzeichnisse erstellt:
o/src/video_processing
o/src/object_detection
o/src/tracking
o/src/gui
o/src/utils
o/tests
•Git-Repository initialisiert
•requirements.txt erstellt mit allen benötigten Paketen
•Pakete installiert und importierbar getestet
•Logging-System konfiguriert
•Basis-Teststruktur (pytest) eingerichtet
•Dummy-Test: Ein Frame aus Video laden und prüfen
Phase 2: Videoverarbeitung & Homography
•Video laden mit OpenCV VideoCapture
•Frames korrekt extrahieren und in numpy speichern
•GUI-Dialog für manuelle Kalibrierung implementiert (6–8 Punkte: Bullypunkte,
Torzentrum)
•Kalibrierungspunkte gespeichert
•Homography-Matrix mit OpenCV findHomography berechnet
•Transformation von Videokoordinaten → Taktikboard-Koordinaten getestet
Phase 3: Objekterkennung
•YOLOv11-Modell vortrainiert eingebunden
•Spieler, Torhüter (optional Ball) erkennen
•Bounding-Boxes und Klassenlabels korrekt zurückgegeben
•Test mit Dummy-Frames: Spieler werden erkannt
Phase 4: Spieler-Tracking
•ByteTrack implementiert
•Spieler-IDs über Frames konsistent•Laufwege für ausgewählte Spieler erstellt
•Test: Laufweg korrekt auf Taktikboard transformiert
Phase 5: Teamzuordnung
•Automatisches Color-Clustering implementiert (HSV-Farbraum, K-Means k=2)
•Spieler korrekt in Team A / Team B gruppiert
•GUI-Korrektur: Trainer kann Spieler-Team wechseln
•Korrekturen für restlichen Clip übernommen
Phase 6: Taktikboard GUI
•GUI-Fenster erstellt (PyQt5/Tkinter)
•Canvas/Board korrekt angezeigt
•Spieler visualisiert:
oEigenes Team: X
oGegner: O
oTorhüter: T
oBall: . (falls erkannt)
•Laufwege für ausgewählte Spieler angezeigt
•Passvorschläge (regelbasiert) visualisiert
•Interaktive Zeichenfunktionen implementiert (J+S Symbole: Pass, Schuss, Laufweg)
•Trainer kann Spieler/Laufwege auswählen, verschieben, korrigieren
Phase 7: Datenexport
•Screenshot / Bild exportieren (PNG/PDF) funktioniert
•Video mit Overlay exportieren (cv2.VideoWriter) funktioniert
•Temporäre Anzeige ohne Speicherung korrekt möglich
Phase 8: Integration & Tests
•End-to-End-Datenfluss: Video → YOLO → Tracking → Team → GUI → Export
•Workflow-Test: Clip laden, Spieler erkennen, Laufwege, Passvorschläge, Export
•Fehlerhandling implementiert:
•
oFalsche Teamzuordnung
oTracking-Ausfälle
oBall nicht erkannt
GUI-Warnungen/Korrekturen getestet•
End-to-End-Integration stabil
Optional / Stretch Goals
•Ballerkennung verbessern (falls klein/unscharf)
•Pose Estimation für Spielerorientierung integrieren
•Mehrere Spieler gleichzeitig tracken
•Automatische Szenenerkennung im Video
•Komplexe Spielzüge vorschlagen (KI-basiert)
•Statistische Auswertungen: Pässe, Schüsse, Positionen
•Tablet/Touchboard-Interaktion optimieren