# Fisheye / GoPro: Undistortion vor Homographie

**Status:** Konzept — **keine Implementierung** ohne ausdrückliche Freigabe.

---

## 1. Problemstellung

- Die **Homographie** modelliert nur eine **projektive Abbildung zwischen zwei Ebenen** (Spielfläche im Bild ↔ Taktikboard-Ebene). Sie setzt voraus, dass die Kamera **zentrale Projektion** nutzt — bei **starker Linsenverzeichnung (Fisheye)** sind Geraden im Raum **nicht mehr gerade im Bild**; Kanten der Banden erscheinen als **krumme Bögen**.
- Ergebnis: Selbst mit vielen Punkt-/Linien-Paaren bleibt ein **systematischer Restfehler**; die Homographie „biegt“ nicht die Verzeichnung aus.

**Ziel:** Das Video (oder zumindest die für Kalibrierung und Tracking genutzten Frames) **zuerst geometrisch entzerren** (Undistortion / Entzerrung), **danach** wie bisher die Homographie auf dem **entzerrten** Bild berechnen und anwenden.

---

## 2. Grundprinzip der Pipeline

Reihenfolge (logisch):

1. **Rohframe** (GoPro, Fisheye).
2. **Undistortion** mit **intrinsischer Kamerakalibrierung** (Matrix \(K\) + Verzeichnungsparameter — Modell siehe Abschnitt 3).
3. **Homographie** nur noch zwischen „nahezu pinhole“-Bild und Board — entspricht der aktuellen UniVision2Board-Logik.

Die Homographie-JSON (`src_points`/`dst_points` oder `src_lines`/`dst_lines`) bezieht sich dann auf **Bildkoordinaten nach Undistortion**. Rohvideo und Export müssen **einheitlich** sein: entweder durchgängig entzerrte Frames verarbeiten oder klar dokumentierte zwei Modi („nur Homographie“ vs. „Undistortion + Homographie“).

---

## 3. Kameramodelle (kurz)

| Ansatz | Eignung GoPro/Fisheye | Hinweis |
|--------|------------------------|---------|
| **Brown-Conrady / „klassisch“** (`cv::initUndistortRectifyMap` mit `k1,k2,p1,p2,k3`) | Nur bei **leichter** Weitwinkel-Verzeichnung oft noch okay | Bei **sehr** großem Öffnungswinkel meist unzureichend |
| **Fisheye-Modell** (OpenCV `fisheye::*`, z. B. Kannala–Brandt-ähnlich) | Typisch für **Action-Cams / stark Fisheye** | Eigene Kalibrier-Routine und andere Parameter als beim Pinhole-Modell |

**Empfehlung für GoPro:** Zuerst **Fisheye-Kalibrierung** evaluieren; falls Restfehler klein genug, kann optional auf das klassische Modell vereinfacht werden — das ist eher eine zweite Mess-/Validierungsphase.

---

## 4. Intrinsische Kalibrierung (offline, einmal pro Kamera/Setup)

- **Eingabe:** Video oder Bildserie mit **Kalibriermuster** (Schachbrett, asymmetrisches Gitter, Charuco — je nach OpenCV-Workflow).
- **Ausgabe:** Datei (z. B. JSON oder `.npz`-Äquivalent), enthält mindestens:
  - **`camera_matrix`** \(K\) (3×3),
  - **Verzeichnungskoeffizienten** (Anzahl und Bedeutung je nach Modell),
  - optional **Bildgröße** \(w,h\) zum Zeitpunkt der Kalibrierung.

**Wichtig:** Wenn später Videos mit **anderer Auflösung** oder **Crop/Lens-Modus** der GoPro kommen, sind \(K\) und Distortion **nicht ohne Weiteres** gültig — Skalierung oder Neukalibrierung nötig.

---

## 5. Einbindung in die bestehende Software (konzeptionell)

### 5.1 Wo Undistortion ausgeführt wird

- **Variante A (empfohlen für Konsistenz):** Ein **einheitlicher Vorverarbeitungsschritt** nach dem Lesen jedes Frames (oder nach Dekodierung im VideoLoader-Pfad), bevor Detection/Tracking/Homographie greifen.
- **Variante B:** Nur für die **Kalibrierungs-GUI** und gespeicherte Referenzframes — **riskant**, wenn das Tracking weiter Rohframes nutzt → zwei verschiedene Geometrien.

Kalibrierung und Laufzeit sollten **dieselbe** Undistortion nutzen.

### 5.2 Speicherung / Konfiguration

- Neue Artefakte z. B. unter `output/` oder Projektconfig:
  - `camera_intrinsics.json` (oder Namenskonvention an Kalibrierungs-Workflow angepasst).
- Pipeline-CLI: optional **`--camera-calib <pfad>`** oder Eintrag in einer bestehenden Config — **ohne** Pfad verhält sich das System wie heute (nur Homographie).

### 5.3 Performance

- Pro Frame: `remap` über **vorab berechnete Maps** (`initUndistortRectifyMap` / fisheye-Äquivalent) ist üblich und GPU-/CPU-lastig aber **deterministisch**.
- Für Echtzeit: Auflösung reduzieren oder Undistortion nur auf **ROI** (Spielfeld-Crop) — optional später optimieren.

---

## 6. Validierung

- **Visuell:** Geraden auf dem Spielfeld (Banden) sollten nach Undistortion **gerader** wirken; dann Homographie-Vorschau im Dialog stabiler.
- **Metrisch:** Reprojektionsfehler aus der **intrinsischen** Kalibrierung dokumentieren; nach Homographie zusätzlich mittlere Abweichung der Rand-Punkte.

---

## 7. Risiken und offene Punkte

- **Zeitaufwand** Nutzer:in: separates Kalibrier-Setup mit Muster.
- **Verschiedene GoPro-Profile** (SuperView, Linear im Body): können sich wie andere „virtuelle Kameras“ verhalten — ggf. **pro Projekt** eine intrinsische Datei.
- **Homographie allein** nach Undistortion löst keine **rolling-shutter**- oder **nicht-planaren** Spielfeld-Probleme — außerhalb dieses Dokuments.

---

## 8. Nächster Schritt nach Freigabe

Erst nach **schriftlicher Bestätigung** (z. B. „Plan fisheye_undistortion umsetzen“): konkrete Bibliothekswahl (OpenCV `fisheye` vs. Standard), Datenformat, Integrationspunkte in `VideoLoader`/`run_pipeline`/Kalibrierungsdialog — als eigene Implementierungsphasen planen.
