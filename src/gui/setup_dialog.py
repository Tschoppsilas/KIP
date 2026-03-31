"""Setup-Dialog für Pipeline-Start (Tkinter).

Schritt 1: Kalibrierung – 6 Feldpunkte anklicken.
Schritt 2: Farb-Seeding – je 2-3 Spieler pro Team + optional Schiri anklicken.

Rückgabe:
    CalibResult  mit src_pts, dst_pts
    SeedResult   mit features_a, features_b, features_ref
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk


# ---------------------------------------------------------------------------
# Feldpunkte (Kalibrierung)
# ---------------------------------------------------------------------------
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
CALIB_COLORS = [
    (255,  80,   0),
    (255, 200,   0),
    (0,   255, 100),
    (0,   220,   0),
    (0,   180, 255),
    (80,   80, 255),
]

# Farben für Team-Seeding
SEED_COLORS = {
    "A":   (220,  40,  40),   # Rot
    "B":   ( 30,  90, 220),   # Blau
    "REF": (160, 160, 160),   # Grau
}

SEED_PATCH = 60   # Pixel-Radius für HSV-Extraktion beim Klick


# ---------------------------------------------------------------------------
# Ergebnis-Dataklassen
# ---------------------------------------------------------------------------
@dataclass
class CalibResult:
    src_pts: list[tuple[float, float]]
    dst_pts: list[tuple[float, float]] = field(
        default_factory=lambda: list(FIELD_COORDS))


@dataclass
class SeedResult:
    features_a:   list[np.ndarray] = field(default_factory=list)
    features_b:   list[np.ndarray] = field(default_factory=list)
    features_ref: list[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Setup-Dialog
# ---------------------------------------------------------------------------
class SetupDialog:
    """Zweischrittiger Setup-Dialog: Kalibrierung + Farb-Seeding.

    Args:
        frame_bgr:  Erster Video-Frame (BGR).
        max_w:      Maximale Fensterbreite in Pixeln.
    """

    STEPS = ["calib", "seed"]

    def __init__(self, frame_bgr: np.ndarray, max_w: int = 1280) -> None:
        self._frame_bgr = frame_bgr
        h, w = frame_bgr.shape[:2]
        self._scale = min(1.0, max_w / w)
        self._dw = int(w * self._scale)
        self._dh = int(h * self._scale)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._base = Image.fromarray(frame_rgb).resize(
            (self._dw, self._dh), Image.LANCZOS)

        self._calib_pts:  list[tuple[int, int]] = []
        self._seed_mode:  str = "A"          # "A", "B", "REF"
        self._seed_pts:   dict[str, list[tuple[int, int]]] = {"A": [], "B": [], "REF": []}

        self._step = "calib"
        self._done = False
        self._photo: ImageTk.PhotoImage | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI-Aufbau
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._root = tk.Tk()
        self._root.title("UniVision2Board – Setup")

        # Canvas
        self._canvas = tk.Label(self._root)
        self._canvas.pack()
        self._canvas.bind("<Button-1>", self._on_click)

        # Status-Bar
        self._status = tk.StringVar()
        tk.Label(self._root, textvariable=self._status,
                 bg="#111", fg="#0f0", font=("Monospace", 11),
                 anchor="w", padx=8).pack(fill=tk.X)

        # Button-Leiste
        btn_frame = tk.Frame(self._root, bg="#222", pady=4)
        btn_frame.pack(fill=tk.X)

        # Kalibrierung: Reset + Weiter
        self._btn_reset = tk.Button(btn_frame, text="↺ Reset",
                                    command=self._reset, width=10)
        self._btn_reset.pack(side=tk.LEFT, padx=6)

        self._btn_next = tk.Button(btn_frame, text="Weiter →",
                                   command=self._next_step,
                                   state=tk.DISABLED, width=14,
                                   bg="#2a2", fg="white")
        self._btn_next.pack(side=tk.LEFT, padx=6)

        # Seeding-Buttons (erst im Schritt 2 sichtbar)
        self._seed_btns: dict[str, tk.Button] = {}
        for key, color in [("A", "#c02020"), ("B", "#1050c0"), ("REF", "#808080")]:
            btn = tk.Button(btn_frame,
                            text=f"{'Team ' + key if key != 'REF' else 'Schiri'}",
                            command=lambda k=key: self._set_seed_mode(k),
                            width=10, relief=tk.RAISED)
            btn.pack(side=tk.LEFT, padx=4)
            btn.pack_forget()
            self._seed_btns[key] = btn

        self._btn_start = tk.Button(btn_frame, text="▶ Pipeline starten",
                                    command=self._finish,
                                    state=tk.DISABLED, width=18,
                                    bg="#186", fg="white")
        self._btn_start.pack(side=tk.RIGHT, padx=8)
        self._btn_start.pack_forget()

        self._root.bind("<Key>", self._on_key)
        self._refresh()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        img = self._base.copy()
        draw = ImageDraw.Draw(img)

        if self._step == "calib":
            self._render_calib(draw)
        else:
            self._render_seed(draw)

        self._photo = ImageTk.PhotoImage(img)
        self._canvas.config(image=self._photo)

    def _render_calib(self, draw: ImageDraw.ImageDraw) -> None:
        n = len(self._calib_pts)
        # Nächsten Punkt anzeigen
        if n < len(FIELD_NAMES):
            draw.rectangle([(0, 0), (self._dw, 46)], fill=(0, 0, 0))
            draw.text((12, 10),
                      f"Schritt 1/2 – Kalibrierung | Punkt {n+1}/{len(FIELD_NAMES)}: {FIELD_NAMES[n]}",
                      fill=CALIB_COLORS[n])
            self._status.set(f"Klicke: {FIELD_NAMES[n]}  →  {FIELD_COORDS[n]}   |  R=Reset")
        else:
            draw.rectangle([(0, 0), (self._dw, 46)], fill=(0, 60, 0))
            draw.text((12, 10),
                      "Alle 6 Punkte gesetzt – 'Weiter' klicken oder Enter drücken",
                      fill=(0, 255, 100))
            self._status.set("Enter = Weiter  |  R = Reset")
            self._btn_next.config(state=tk.NORMAL)

        for i, (px, py) in enumerate(self._calib_pts):
            c = CALIB_COLORS[i]
            r = 8
            draw.ellipse([(px-r, py-r), (px+r, py+r)], fill=c, outline=(0,0,0), width=2)
            draw.text((px+12, py-6), f"{i+1} {FIELD_NAMES[i]}", fill=c)

    def _render_seed(self, draw: ImageDraw.ImageDraw) -> None:
        mode_color = SEED_COLORS[self._seed_mode]
        mode_label = f"Team {self._seed_mode}" if self._seed_mode != "REF" else "Schiri"
        draw.rectangle([(0, 0), (self._dw, 46)], fill=(0, 0, 0))
        draw.text((12, 10),
                  f"Schritt 2/2 – Farb-Seeding | Aktiv: {mode_label}"
                  f"  (A:{len(self._seed_pts['A'])}  B:{len(self._seed_pts['B'])}"
                  f"  Schiri:{len(self._seed_pts['REF'])})",
                  fill=mode_color)
        self._status.set(
            f"Klicke Spieler für {mode_label}  |  Buttons oben zum Wechseln  |  "
            "Mind. 2× Team A + 2× Team B, dann 'Pipeline starten'")

        for key, pts in self._seed_pts.items():
            c = SEED_COLORS[key]
            for px, py in pts:
                r = 12
                draw.ellipse([(px-r, py-r), (px+r, py+r)],
                             outline=c, width=3)
                draw.text((px+14, py-6),
                          f"{'A' if key == 'A' else 'B' if key == 'B' else 'S'}",
                          fill=c)

        # Start-Button aktivieren?
        if len(self._seed_pts["A"]) >= 2 and len(self._seed_pts["B"]) >= 2:
            self._btn_start.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Event-Handler
    # ------------------------------------------------------------------
    def _on_click(self, event: tk.Event) -> None:
        x, y = event.x, event.y
        if self._step == "calib":
            if len(self._calib_pts) < len(FIELD_NAMES):
                self._calib_pts.append((x, y))
        else:
            self._seed_pts[self._seed_mode].append((x, y))
        self._refresh()

    def _on_key(self, event: tk.Event) -> None:
        k = event.keysym
        if k == "Escape":
            self._root.destroy()
        elif k.lower() == "r":
            self._reset()
        elif k == "Return":
            if self._step == "calib" and len(self._calib_pts) >= 4:
                self._next_step()
            elif self._step == "seed":
                self._finish()

    def _reset(self) -> None:
        if self._step == "calib":
            self._calib_pts.clear()
            self._btn_next.config(state=tk.DISABLED)
        else:
            self._seed_pts = {"A": [], "B": [], "REF": []}
            self._btn_start.config(state=tk.DISABLED)
        self._refresh()

    def _next_step(self) -> None:
        self._step = "seed"
        self._btn_next.pack_forget()
        self._btn_reset.config(text="↺ Reset Seeding")
        for btn in self._seed_btns.values():
            btn.pack(side=tk.LEFT, padx=4)
        self._btn_start.pack(side=tk.RIGHT, padx=8)
        self._set_seed_mode("A")
        self._refresh()

    def _set_seed_mode(self, mode: str) -> None:
        self._seed_mode = mode
        for key, btn in self._seed_btns.items():
            btn.config(relief=tk.SUNKEN if key == mode else tk.RAISED)
        self._refresh()

    def _finish(self) -> None:
        self._done = True
        self._root.destroy()

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
    def run(self) -> tuple[CalibResult | None, SeedResult | None]:
        """Zeigt den Dialog und wartet auf Benutzer-Eingabe."""
        self._root.mainloop()

        if not self._done or len(self._calib_pts) < 4:
            return None, None

        # Kalibrierungs-Punkte zurück auf Originalauflösung skalieren
        src_pts = [(x / self._scale, y / self._scale)
                   for x, y in self._calib_pts]
        calib = CalibResult(src_pts=src_pts,
                            dst_pts=list(FIELD_COORDS[:len(src_pts)]))

        # Farb-Features aus geklickten Punkten extrahieren
        seed = SeedResult()
        for key, pts in self._seed_pts.items():
            feat_list = (seed.features_a if key == "A"
                         else seed.features_b if key == "B"
                         else seed.features_ref)
            for px, py in pts:
                # Auf Originalauflösung zurückskalieren
                ox = int(px / self._scale)
                oy = int(py / self._scale)
                h, w = self._frame_bgr.shape[:2]
                r = SEED_PATCH
                y1, y2 = max(0, oy - r), min(h, oy + r)
                x1, x2 = max(0, ox - r), min(w, ox + r)
                crop = self._frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                feat = hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)
                feat_list.append(feat)

        return calib, seed
