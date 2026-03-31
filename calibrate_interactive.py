"""Interaktives Kalibrierungs-Tool – Tkinter-basiert (kein OpenCV-GUI noetig).

Ablauf:
  1. Erster Frame aus dem Video wird in einem Tkinter-Fenster angezeigt
  2. Du klickst nacheinander auf 6 Spielfeldpunkte
  3. Homography wird berechnet
  4. Ergebnis wird als PNG gespeichert + Kalibrierung als JSON

Aufruf:
    ./.venv/bin/python calibrate_interactive.py Videos/Trainingsdaten/Mittelland_4.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk

from video_processing.video_reader import read_first_frame, get_video_info
from video_processing.homography import compute_homography, transform_points
from video_processing.calibration import save_calibration

VIDEO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Videos/Muenchenstein_1.mp4")

# Unihockey-Feldpunkte in Reihenfolge (m)
FIELD_NAMES = [
    "Ecke oben-links",
    "Ecke oben-rechts",
    "Ecke unten-rechts",
    "Ecke unten-links",
    "Mittellinie oben",
    "Mittellinie unten",
]
FIELD_COORDS = [
    (0.0,  0.0),
    (40.0, 0.0),
    (40.0, 20.0),
    (0.0,  20.0),
    (20.0, 0.0),
    (20.0, 20.0),
]
COLORS_RGB = [
    (255,  80,   0),
    (255, 200,   0),
    (0,   255, 100),
    (0,   255,   0),
    (0,   180, 255),
    (80,   80, 255),
]
N_POINTS = len(FIELD_NAMES)

# ─── Video laden ─────────────────────────────────────────────────────────────
print(f"\nLade Video: {VIDEO}")
info  = get_video_info(VIDEO)
frame_bgr = read_first_frame(VIDEO)
print(f"Aufloesung: {info.width}x{info.height} | {info.fps} fps | {info.frame_count} Frames")

MAX_W = 1280
scale = min(1.0, MAX_W / info.width)
disp_w = int(info.width  * scale)
disp_h = int(info.height * scale)

frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
frame_pil = Image.fromarray(frame_rgb).resize((disp_w, disp_h), Image.LANCZOS)

clicked: list[tuple[int, int]] = []


def _render() -> ImageTk.PhotoImage:
    canvas_img = frame_pil.copy()
    draw = ImageDraw.Draw(canvas_img)
    next_idx = len(clicked)

    # Statustext oben
    if next_idx < N_POINTS:
        name  = FIELD_NAMES[next_idx]
        color = COLORS_RGB[next_idx]
        draw.rectangle([(0, 0), (disp_w, 48)], fill=(0, 0, 0, 180))
        draw.text((14, 10), f"Klicke Punkt {next_idx+1}/{N_POINTS}: {name}",
                  fill=color)
    else:
        draw.rectangle([(0, 0), (disp_w, 48)], fill=(0, 0, 0, 180))
        draw.text((14, 10),
                  "Alle Punkte gesetzt – Enter zum Bestaetigen  |  R = Reset  |  ESC = Abbruch",
                  fill=(0, 255, 100))

    # Gesetzte Punkte
    for i, (px, py) in enumerate(clicked):
        c = COLORS_RGB[i]
        r = 9
        draw.ellipse([(px-r, py-r), (px+r, py+r)], fill=c, outline=(0,0,0), width=2)
        draw.text((px + 13, py - 7), f"{i+1} {FIELD_NAMES[i]}", fill=c)

    return ImageTk.PhotoImage(canvas_img)


# ─── Tkinter-Fenster ─────────────────────────────────────────────────────────
root = tk.Tk()
root.title("UniVision2Board – Kalibrierung (Enter=OK | R=Reset | ESC=Abbruch)")

lbl = tk.Label(root)
lbl.pack()

status_var = tk.StringVar(value=f"Klicke Punkt 1/{N_POINTS}: {FIELD_NAMES[0]}")
status_bar = tk.Label(root, textvariable=status_var, bg="#222", fg="#0f0",
                      font=("Monospace", 11), anchor="w", padx=8)
status_bar.pack(fill=tk.X)

_photo_ref = None  # Referenz halten damit GC nicht löscht


def _refresh():
    global _photo_ref
    _photo_ref = _render()
    lbl.config(image=_photo_ref)
    n = len(clicked)
    if n < N_POINTS:
        status_var.set(f"Klicke Punkt {n+1}/{N_POINTS}: {FIELD_NAMES[n]}")
    else:
        status_var.set("Alle Punkte gesetzt – Enter=Bestaetigen  R=Reset  ESC=Abbruch")


def _on_click(event):
    if len(clicked) < N_POINTS:
        x, y = event.x, event.y
        clicked.append((x, y))
        print(f"  Punkt {len(clicked)}: ({x}, {y})  →  {FIELD_NAMES[len(clicked)-1]}")
        _refresh()


def _on_key(event):
    key = event.keysym
    if key == "Escape":
        print("Abgebrochen.")
        root.destroy()
    elif key.lower() == "r":
        clicked.clear()
        print("Punkte zurueckgesetzt.")
        _refresh()
    elif key == "Return" and len(clicked) >= 4:
        _confirm()


def _confirm():
    root.destroy()


lbl.bind("<Button-1>", _on_click)
root.bind("<Key>", _on_key)

print(f"\nFenster geoeffnet – klicke {N_POINTS} Punkte in dieser Reihenfolge:")
for i, n in enumerate(FIELD_NAMES):
    print(f"  {i+1}. {n}  →  {FIELD_COORDS[i]}")
print("\nEnter = bestaetigen (ab 4 Punkten) | R = zuruecksetzen | ESC = abbrechen\n")

_refresh()
root.mainloop()

if len(clicked) < 4:
    print("Zu wenige Punkte – Abbruch.")
    sys.exit(0)

# ─── Homography berechnen ─────────────────────────────────────────────────────
n = len(clicked)
src_px = [(x / scale, y / scale) for x, y in clicked]
dst_m  = list(FIELD_COORDS[:n])

H = compute_homography(src_px, dst_m)
print(f"\nHomography-Matrix:\n{H}\n")

from video_processing.homography import transform_point
cx, cy = transform_point((info.width / 2, info.height / 2), H)
print(f"Bildmitte ({info.width//2}, {info.height//2}) → Feld ({cx:.2f} m, {cy:.2f} m)")

# ─── Ergebnis-Bild speichern ──────────────────────────────────────────────────
result_bgr = frame_bgr.copy()
H_inv = np.linalg.inv(H)
for x in range(0, 41, 5):
    pts = [(float(x), float(y)) for y in np.linspace(0, 20, 30)]
    for gx, gy in transform_points(pts, H_inv):
        if 0 <= gx <= info.width and 0 <= gy <= info.height:
            cv2.circle(result_bgr, (int(gx), int(gy)), 2, (0, 220, 255), -1)
for y in range(0, 21, 5):
    pts = [(float(x), float(y)) for x in np.linspace(0, 40, 50)]
    for gx, gy in transform_points(pts, H_inv):
        if 0 <= gx <= info.width and 0 <= gy <= info.height:
            cv2.circle(result_bgr, (int(gx), int(gy)), 2, (0, 220, 255), -1)

for i, (px, py) in enumerate(src_px):
    c_bgr = tuple(reversed(COLORS_RGB[i]))
    cv2.circle(result_bgr, (int(px), int(py)), 14, (0,0,0), -1)
    cv2.circle(result_bgr, (int(px), int(py)), 11, c_bgr, -1)
    label = f"{i+1}: {FIELD_NAMES[i]}"
    cv2.putText(result_bgr, label, (int(px)+16, int(py)+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(result_bgr, label, (int(px)+16, int(py)+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, c_bgr, 2, cv2.LINE_AA)

out_png = Path("calibration_result.png")
cv2.imwrite(str(out_png), result_bgr)
print(f"Ergebnis-Bild gespeichert: {out_png.resolve()}")

# ─── Kalibrierung als JSON speichern ─────────────────────────────────────────
stem = VIDEO.stem.lower().replace(" ", "_")
calib_path = Path(f"calibration_{stem}.json")
save_calibration(src_px, dst_m, calib_path,
                 metadata={"video": VIDEO.name, "field_width_m": 40, "field_height_m": 20})
print(f"Kalibrierung gespeichert:  {calib_path.resolve()}")
print("\nFertig! Oeffne calibration_result.png um das Ergebnis zu pruefen.")
