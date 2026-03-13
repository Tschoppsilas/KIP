Programmiersprache
•
Python – weit verbreitet, sehr gute Bibliotheken für Computer Vision, Tracking, KI,
Videoverarbeitung und Visualisierung.
Videoverarbeitung
•
OpenCV → Laden, Abspielen und Bearbeiten von Videos, Extrahieren einzelner Frames,
Transformation des Spielfelds via Homography, Zeichnen auf dem Taktikboard.
Objekterkennung
•
YOLOv11 (vortrainiert) → Erkennung von Spielern, Torhütern und (optional) Ball. Liefert
Bounding-Boxes und Klassenlabels.
Spieler-Tracking
•
ByteTrack → Multi-Object-Tracking für Spieler-Bounding-Boxes über Frames hinweg.
Liefert ID-Konsistenz über Zeit und ermöglicht Laufwege.
Teamzuordnung
•Color-Clustering (z. B. K-Means in HSV-Farbraum) → Gruppiert Spieler automatisch in
zwei Teams anhand der Trikotfarbe.
•Trainer-Korrektur → GUI-Interaktion, um fehlerhafte Zuordnungen zu korrigieren.
Taktikboard / Visualisierung
•PyQt5 oder Tkinter → GUI für Desktop-Anwendung
•Touch-Board / Tablet-Interface → Darstellung der Spieler (X / O), Torhüter (T), Ball (.),
Laufwege und Passvorschläge.
•Interaktive Zeichenfunktionen → J+S-konforme Symbole für Pässe, Schüsse und
Laufwege.
Passvorschläge
•
Regelbasierte Logik → Sichtbare Mitspieler erkennen, durch Gegner blockierte Pässe
ausschließen, Pfeile auf dem Taktikboard einzeichnen.
Datenexport
•Screenshot / Bild → PNG/PDF
•Videosequenz mit Overlay → OpenCV kann Frames mit Taktikboard-Darstellung
speichern und zu Video zusammensetzen.
Damit haben wir ein komplettes MVP-System, das:
•Spieler erkennt und optional Ball/Torhüter trackt
•Laufwege visualisiert
•Passvorschläge regelbasiert anzeigt
•Trainer-Interaktion erlaubt (Auswahl, Korrektur, Zeichnen)
•Szenen als Bild oder Video exportiert