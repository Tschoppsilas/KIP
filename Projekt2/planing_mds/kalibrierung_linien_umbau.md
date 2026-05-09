# Kalibrierungsdialog: Umbau von Punkt-Paaren auf Linien-Paare

**Status:** Plan umgesetzt (Schritte 1–5; Stand siehe `homography.py`, `calibration_dialog.py`, `tests/test_video_processing.py`, `planing_mds/workflow_test.md`).  
**Priorität:** Gezielte Verbesserung der Kalibrierung (**SHOULD**), **kein MVP-MUST**.

---

## 1. Ziel

Der Trainer soll die **sichtbaren Banden** nicht mehr nur durch wenige Einzelpunkte abbilden, sondern durch **vier zugeordnete Kantenlinien** (Video ↔ Taktikboard). Entlang dieser Paare werden **viele virtuelle Punkt-Korrespondenzen** erzeugt (Sampling), sodass `cv2.findHomography` mit **dicht verteilten Randlagen** arbeitet — analog zur bestehenden Pipeline in `HomographyTransformer`, aber mit robusterer geometrischer Abdeckung der Außenlinien.

---

## 2. Grundprinzip (unveränderte Mathematik)

- Es bleibt eine **projektive Homographie** zwischen Bildebene (Video) und Board-Ebene.
- OpenCV: weiterhin **`cv2.findHomography`** mit Punktlisten `src` / `dst` (ggf. RANSAC-Parameter feinjustieren bei vielen Punkten).
- Neu ist nur die **Generierung** dieser Listen: aus **4 Linien-Paaren** → **N sampled Punkt-Paare** pro Linie (oder feste Anzahl insgesamt), dann wie bisher `compute`.

---

## 3. Funktionaler Workflow (Zielbild)

1. **Vier Kanten** festlegen (Reihenfolge eindeutig dokumentiert, z. B. oben → rechts → unten → links im **sichtbaren** Feld).
2. Pro Kante nacheinander (oder in einem klaren Schrittmodus):
   - Im **Video**: Linie zeichnen (Minimalvariante: **Start- und Endpunkt** einer Strecke entlang der Bande; optional später: Polylinie).
   - Im **Board**: dieselbe **physische** Kante mit Start/Ende markieren.
3. Nach **vollständigen 4 Paaren**: Homographie aus **Samples** berechnen, **Live-Vorschau** wie heute (`warpPerspective` + Blend).
4. **Speichern / Laden**: siehe Abschnitt 6.

Abbruch, Undo und Zoom/Scroll bleiben als UX-Ziele erhalten (konkrete UI-Details erst bei Implementierung gemäß diesem Plan).

---

## 4. Betroffene Dateien (voraussichtlich)

| Datei | Rolle |
|--------|--------|
| `univision2board/src/gui/calibration_dialog.py` | Hauptumbau: Zustand von Punkt-Listen → Linien-Paare; Zeichen-/Klicklogik; Overlays (Linien statt Punktnummern); Statuszeilen; Vorschau-Trigger |
| `univision2board/src/video_processing/homography.py` | Erweiterung: z. B. `compute_from_line_pairs(...)` oder Hilfsfunktion **Sampling** → ruft intern bestehendes `compute(src_points, dst_points)` auf; optional zweites JSON-Schema oder Adapter beim Laden |
| `univision2board/tests/test_video_processing.py` | Tests für neue Hilfsfunktionen (Sampling, Mindestanzahl Linien, Konsistenz Matrix) |
| `univision2board/tests/test_e2e.py` / weitere Tests | Anpassung nur, falls Kalibrierungs-JSON-Format oder Fixtures sich ändern |
| `univision2board/run_pipeline.py` | Nur falls sich Signaturen oder Ladepfade der Kalibrierung ändern (minimal halten) |

**Nicht zwingend:** neue Datei `calibration_lines.py` oder `homography_lines.py` nur, wenn die Logik aus `calibration_dialog.py` heraus verlagert werden soll (Lesbarkeit).

---

## 5. Was sich konkret ändert

### 5.1 Datenmodell im Dialog

- **Bisher:** `List[Tuple[float,float]]` für `_src_points` / `_dst_points`, synchron als Paare.
- **Neu:** Vier Einträge je Ansicht, z. B. `List[Tuple[Tuple[float,float], Tuple[float,float]]]` (Start/Ende je Linie in Bildkoordinaten), oder äquivalente Struktur mit klarer Kanten-ID `0..3`.

### 5.2 Hilfslinien / Guides

- **`apply_video_field_guide` / `apply_board_field_guide`**: Anpassung von „8 Punkten“ auf **vier empfohlene Kanten** (schematisches Rechteck + Beschriftung 1–4), falls weiterhin Overlays gewünscht.
- **`PAIR_COLORS`**: auf **4 Kantenfarben** reduzieren oder beibehalten für zusätzliche UI-Elemente.

### 5.3 Homographie-Berechnung

- Neue rein **numerische** Schicht (ideal in `homography.py`):
  - Eingabe: 4 Linien-Paare (Video/Board).
  - Ausgabe: zwei große Listen `src_points`, `dst_points` durch **lineares Sampling** auf jeder Strecke (Parameter \(t \in [0,1]\), gleiche Anzahl Stützstellen pro Linie oder gesamt begrenzt).
  - Aufruf: `HomographyTransformer.compute(src_points, dst_points)` unverändert — **keine Duplikation** der `findHomography`-Logik.

### 5.4 UI-Interaktion

- Ersetzen der „Punkt klicken → nächster Punkt auf Board“-Schleife durch einen **Linien-Workflow** (z. B. zwei Klicks pro Linie und Ansicht; oder Drag — nur wenn im Plan bei Implementierung festgehalten).
- **`_ClickableImage`** oder Nachfolger: Mausereignisse für **zwei Punkte pro Linie**, Zeichnen der Strecke, ggf. Shift/Winkel optional später.

### 5.5 Konstanten

- `MIN_POINTS` / `MAX_POINTS` entfallen oder werden durch **`NUM_LINES = 4`** und Sampling-Parameter (`SAMPLES_PER_LINE` oder `TOTAL_SAMPLES`) ersetzt.

---

## 6. Kalibrierungs-JSON (Kompatibilität)

**Ziel:** Alte Kalibrierungen weiter nutzbar oder klare Migration.

**Option A (empfohlen für Übergang):**

- JSON erhält Feld **`"schema_version"`** oder **`"mode"`**: `"points"` | `"lines"`.
- **`lines`**: Speicherung der 4 Strecken als `[[x1,y1],[x2,y2]]` je Bild.
- Beim Laden: wenn `"points"` → bisheriger Pfad; wenn `"lines"` → Sampling → `compute`.

**Option B:**

- Nur neues Format; alter Loader wandelt alte `src_points`/`dst_points` mit Hinweisdialog („Legacy“).

Die konkrete Option wird **bei Implementierung** gemäß diesem Dokument festgelegt und hier nicht weiter verzweigt — wichtig ist **ein** konsistenter Ansatz und Tests.

---

## 7. Implementierungsschritte (Reihenfolge bei Freigabe)

1. **Sampling + Tests** in `homography.py` (kein GUI): feste Eingabelinien → erwartete Matrix-Eigenschaften / Reprojektionsfehler-Sanity.
2. **JSON-Schema / Laden-Speichern** in `HomographyTransformer` oder dediziertem Modul, inkl. optionaler Rückwärtskompatibilität.
3. **Refaktor `calibration_dialog.py`**: Zustand, Guides, Zeichnen, Vorschau, Speichern.
4. **Manuelle GUI-Prüfung** (nach wie vor ggf. separates Testskript / ignorierte Tests ohne Display).
5. **Dokumentation:** kurzer Eintrag in `planing_mds/workflow_test.md` oder `todo.md`, falls ihr dort Workflows pflegt.

---

## 8. Nicht-Ziele (für diesen Umbau)

- Keine **Kamera-Kalibrierung** / Entzerrung von Linsenverzeichnung (bleibt außerhalb Homographie).
- Kein **automatisches** Kantendetektieren im Video — weiterhin manuelle Markierung.

---

## 9. Abnahme

Umsetzung startet **erst nach deiner schriftlichen Bestätigung** (z. B. „Plan umsetzen wie in `planning_mds/kalibrierung_linien_umbau.md`“). Abweichungen vom Plan nur nach expliziter Nachjustierung dieses Dokuments.
