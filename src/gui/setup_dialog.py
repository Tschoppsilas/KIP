"""Setup-Dialog für Pipeline-Start (Tkinter).

Schritt 1 – Kalibrierung:   6 Feldpunkte im Video-Frame anklicken.
Schritt 2 – Vorschau:       Kalibrierung auf dem Taktikboard prüfen.
Schritt 3 – Farb-Seeding:   je 2-3 Spieler pro Team + Schiri anklicken.

Rückgabe:
    CalibResult  mit src_pts, dst_pts
    SeedResult   mit features_a, features_b, features_ref
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

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
    (0,   200,  80),
    (0,   220,   0),
    (0,   180, 255),
    (80,   80, 255),
]

SEED_COLORS = {
    "A":   (220,  40,  40),
    "B":   ( 30,  90, 220),
    "REF": (160, 160, 160),
}

SEED_PATCH = 60

BOARD_PATH = Path(__file__).parent.parent.parent / "Taktikboard" / "Taktikboard.png"
BOARD_FIELD_W = 40.0   # Meter
BOARD_FIELD_H = 20.0   # Meter


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
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _field_to_board(pt: tuple[float, float],
                    board_w: int, board_h: int) -> tuple[int, int]:
    """Feld-Koordinaten (Meter) → Pixel-Koordinaten auf dem Taktikboard."""
    # Das Spielfeld belegt das Bild vollständig (Rand berücksichtigt)
    margin_x = int(board_w * 0.04)
    margin_y = int(board_h * 0.08)
    fw = board_w - 2 * margin_x
    fh = board_h - 2 * margin_y
    px = margin_x + int(pt[0] / BOARD_FIELD_W * fw)
    py = margin_y + int(pt[1] / BOARD_FIELD_H * fh)
    return px, py


# ---------------------------------------------------------------------------
# Setup-Dialog (3 Schritte)
# ---------------------------------------------------------------------------
class SetupDialog:
    """Dreischrittiger Setup-Dialog: Kalibrierung → Board-Vorschau → Seeding.

    Args:
        frame_bgr:   Erster Video-Frame (BGR).
        board_path:  Pfad zum Taktikboard-PNG.
        max_w:       Maximale Fensterbreite in Pixeln.
    """

    def __init__(
        self,
        frame_bgr: np.ndarray,
        board_path: Path = BOARD_PATH,
        max_w: int = 1280,
    ) -> None:
        self._frame_bgr = frame_bgr
        h, w = frame_bgr.shape[:2]
        self._scale = min(1.0, max_w / w)
        self._dw = int(w * self._scale)
        self._dh = int(h * self._scale)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._video_img = Image.fromarray(frame_rgb).resize(
            (self._dw, self._dh), Image.LANCZOS)

        # Taktikboard laden + auf gleiche Grösse skalieren
        if board_path.exists():
            board = Image.open(board_path).convert("RGB")
            self._board_img = board.resize((self._dw, self._dh), Image.LANCZOS)
            self._board_w, self._board_h = self._dw, self._dh
        else:
            # Fallback: grünes Rechteck
            self._board_img = Image.new("RGB", (self._dw, self._dh), (0, 120, 0))
            self._board_w, self._board_h = self._dw, self._dh

        self._calib_pts:  list[tuple[int, int]] = []
        self._seed_mode:  str = "A"
        self._seed_pts:   dict[str, list[tuple[int, int]]] = {
            "A": [], "B": [], "REF": []}

        # Board-Punkte: werden bei Schritt-2-Eintritt initialisiert,
        # dann per Drag & Drop anpassbar
        self._board_pts:  list[tuple[int, int]] = []
        self._drag_idx:   int = -1   # Index des gerade gezogenen Punktes

        self._step = "calib"   # "calib" → "board" → "seed"
        self._done = False
        self._photo: ImageTk.PhotoImage | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI-Aufbau
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._root = tk.Tk()
        self._root.title("UniVision2Board – Setup")
        self._root.configure(bg="#1a1a1a")

        # Anzeige-Canvas
        self._canvas = tk.Label(self._root, bg="#1a1a1a")
        self._canvas.pack(pady=(4, 0))

        # Status-Leiste
        self._status = tk.StringVar()
        tk.Label(self._root, textvariable=self._status,
                 bg="#111", fg="#0f0", font=("Monospace", 11),
                 anchor="w", padx=8, pady=4).pack(fill=tk.X)

        # Button-Leiste
        self._btn_frame = tk.Frame(self._root, bg="#222", pady=6)
        self._btn_frame.pack(fill=tk.X)

        # --- Schritt 1: Kalibrierung ---
        self._btn_reset = tk.Button(
            self._btn_frame, text="↺ Reset", command=self._reset,
            width=10, bg="#444", fg="white")
        self._btn_reset.pack(side=tk.LEFT, padx=6)

        self._btn_next = tk.Button(
            self._btn_frame, text="Weiter →", command=self._go_to_board,
            state=tk.DISABLED, width=16, bg="#2a6", fg="white",
            font=("Arial", 10, "bold"))
        self._btn_next.pack(side=tk.LEFT, padx=6)

        # --- Schritt 2: Board-Vorschau ---
        self._btn_back = tk.Button(
            self._btn_frame, text="← Zurück", command=self._go_back_to_calib,
            width=14, bg="#555", fg="white")
        self._btn_back.pack(side=tk.LEFT, padx=6)
        self._btn_back.pack_forget()

        self._btn_confirm = tk.Button(
            self._btn_frame, text="Kalibrierung OK →", command=self._go_to_seed,
            width=20, bg="#2a6", fg="white",
            font=("Arial", 10, "bold"))
        self._btn_confirm.pack(side=tk.LEFT, padx=6)
        self._btn_confirm.pack_forget()

        # --- Schritt 3: Seeding ---
        self._seed_btns: dict[str, tk.Button] = {}
        for key, bg in [("A", "#c02020"), ("B", "#1050c0"), ("REF", "#606060")]:
            label = f"Team {key}" if key != "REF" else "Schiri"
            btn = tk.Button(
                self._btn_frame, text=label,
                command=lambda k=key: self._set_seed_mode(k),
                width=10, bg=bg, fg="white", relief=tk.RAISED)
            btn.pack(side=tk.LEFT, padx=4)
            btn.pack_forget()
            self._seed_btns[key] = btn

        self._btn_start = tk.Button(
            self._btn_frame, text="▶ Pipeline starten",
            command=self._finish,
            state=tk.DISABLED, width=20, bg="#186", fg="white",
            font=("Arial", 10, "bold"))
        self._btn_start.pack(side=tk.RIGHT, padx=8)
        self._btn_start.pack_forget()

        self._root.bind("<Key>", self._on_key)
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",        self._on_drag)
        self._canvas.bind("<ButtonRelease-1>",  self._on_release)
        self._refresh()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        if self._step == "calib":
            img = self._video_img.copy()
            self._render_calib(ImageDraw.Draw(img))
        elif self._step == "board":
            img = self._render_board_preview()
        else:
            img = self._video_img.copy()
            self._render_seed(ImageDraw.Draw(img))

        self._photo = ImageTk.PhotoImage(img)
        self._canvas.config(image=self._photo)

    def _render_calib(self, draw: ImageDraw.ImageDraw) -> None:
        n = len(self._calib_pts)
        if n < len(FIELD_NAMES):
            draw.rectangle([(0, 0), (self._dw, 48)], fill=(0, 0, 0))
            draw.text((12, 12),
                      f"Schritt 1/3 – Kalibrierung | "
                      f"Punkt {n+1}/{len(FIELD_NAMES)}: {FIELD_NAMES[n]}",
                      fill=CALIB_COLORS[n])
            self._status.set(
                f"Klicke: {FIELD_NAMES[n]}  →  Feldkoord. {FIELD_COORDS[n]}   "
                "| R = Reset")
            self._btn_next.config(state=tk.DISABLED)
        else:
            draw.rectangle([(0, 0), (self._dw, 48)], fill=(0, 50, 0))
            draw.text((12, 12),
                      "Alle 6 Punkte gesetzt – 'Weiter' klicken oder Enter",
                      fill=(0, 255, 100))
            self._status.set("Enter = Weiter zum Taktikboard  |  R = Reset")
            self._btn_next.config(state=tk.NORMAL)

        for i, (px, py) in enumerate(self._calib_pts):
            c = CALIB_COLORS[i]
            r = 8
            draw.ellipse([(px-r, py-r), (px+r, py+r)],
                         fill=c, outline=(0, 0, 0), width=2)
            draw.text((px + 12, py - 6), f"{i+1} {FIELD_NAMES[i]}", fill=c)

    def _render_board_preview(self) -> Image.Image:
        """Zeigt das Taktikboard mit verschiebbaren Kalibrierungspunkten."""
        img = self._board_img.copy()
        draw = ImageDraw.Draw(img)

        draw.rectangle([(0, 0), (self._dw, 48)], fill=(0, 0, 30))
        draw.text((12, 12),
                  "Schritt 2/3 – Kalibrierung auf Taktikboard prüfen & anpassen",
                  fill=(180, 220, 255))
        self._status.set(
            "Punkte per Drag & Drop verschieben falls nötig  |  "
            "'Kalibrierung OK' wenn alles stimmt  |  '← Zurück' zum Neusetzen")

        for i, (bx, by) in enumerate(self._board_pts):
            c = CALIB_COLORS[i]
            r = 10
            # Punkt mit Highlight wenn gerade gezogen
            outline = (255, 255, 0) if i == self._drag_idx else (255, 255, 255)
            draw.ellipse([(bx-r, by-r), (bx+r, by+r)],
                         fill=c, outline=outline, width=2)
            draw.text((bx + 13, by - 6), f"{i+1} {FIELD_NAMES[i]}", fill=c)

        return img

    def _render_seed(self, draw: ImageDraw.ImageDraw) -> None:
        mode_color = SEED_COLORS[self._seed_mode]
        mode_label = (f"Team {self._seed_mode}"
                      if self._seed_mode != "REF" else "Schiri")
        n_a   = len(self._seed_pts["A"])
        n_b   = len(self._seed_pts["B"])
        n_ref = len(self._seed_pts["REF"])
        draw.rectangle([(0, 0), (self._dw, 48)], fill=(0, 0, 0))
        draw.text((12, 12),
                  f"Schritt 3/3 – Farb-Seeding | Aktiv: {mode_label}"
                  f"   A:{n_a}  B:{n_b}  Schiri:{n_ref}",
                  fill=mode_color)
        self._status.set(
            f"Klicke auf Trikot von {mode_label}  |  "
            "Mind. 2× Team A + 2× Team B, dann 'Pipeline starten'")

        for key, pts in self._seed_pts.items():
            c = SEED_COLORS[key]
            lbl = "A" if key == "A" else ("B" if key == "B" else "S")
            for px, py in pts:
                r = 12
                draw.ellipse([(px-r, py-r), (px+r, py+r)],
                             outline=c, width=3)
                draw.text((px + 14, py - 6), lbl, fill=c)

        if n_a >= 2 and n_b >= 2:
            self._btn_start.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Schritt-Wechsel
    # ------------------------------------------------------------------
    def _go_to_board(self) -> None:
        """Schritt 1 → 2: Kalibrierung → Board-Vorschau mit Drag & Drop."""
        self._step = "board"
        # Board-Punkte aus FIELD_COORDS berechnen (automatische Startposition)
        self._board_pts = [
            _field_to_board(FIELD_COORDS[i], self._board_w, self._board_h)
            for i in range(len(self._calib_pts))
        ]
        self._drag_idx = -1
        self._btn_next.pack_forget()
        self._btn_reset.pack_forget()
        self._btn_back.pack(side=tk.LEFT, padx=6)
        self._btn_confirm.pack(side=tk.LEFT, padx=6)
        self._refresh()

    def _go_back_to_calib(self) -> None:
        """Schritt 2 → 1: zurück zur Kalibrierung."""
        self._step = "calib"
        self._btn_back.pack_forget()
        self._btn_confirm.pack_forget()
        self._btn_reset.pack(side=tk.LEFT, padx=6)
        self._btn_next.pack(side=tk.LEFT, padx=6)
        self._refresh()

    def _go_to_seed(self) -> None:
        """Schritt 2 → 3: Board bestätigt → Farb-Seeding."""
        self._step = "seed"
        self._btn_back.pack_forget()
        self._btn_confirm.pack_forget()
        self._btn_reset.config(text="↺ Reset Seeding")
        self._btn_reset.pack(side=tk.LEFT, padx=6)
        for btn in self._seed_btns.values():
            btn.pack(side=tk.LEFT, padx=4)
        self._btn_start.pack(side=tk.RIGHT, padx=8)
        self._set_seed_mode("A")
        self._refresh()

    # ------------------------------------------------------------------
    # Event-Handler
    # ------------------------------------------------------------------
    def _on_press(self, event: tk.Event) -> None:
        x, y = event.x, event.y
        if self._step == "calib":
            if len(self._calib_pts) < len(FIELD_NAMES):
                self._calib_pts.append((x, y))
            self._refresh()
        elif self._step == "board":
            # Nächsten Punkt zum Verschieben suchen (innerhalb 20px)
            self._drag_idx = -1
            for i, (bx, by) in enumerate(self._board_pts):
                if abs(bx - x) < 20 and abs(by - y) < 20:
                    self._drag_idx = i
                    break
            self._refresh()
        elif self._step == "seed":
            self._seed_pts[self._seed_mode].append((x, y))
            self._refresh()

    def _on_drag(self, event: tk.Event) -> None:
        if self._step == "board" and self._drag_idx >= 0:
            # Punkt verschieben (innerhalb Bildgrenzen)
            x = max(0, min(self._dw, event.x))
            y = max(0, min(self._dh, event.y))
            self._board_pts[self._drag_idx] = (x, y)
            self._refresh()

    def _on_release(self, event: tk.Event) -> None:
        if self._step == "board":
            self._drag_idx = -1
            self._refresh()

    def _on_key(self, event: tk.Event) -> None:
        k = event.keysym
        if k == "Escape":
            self._root.destroy()
        elif k.lower() == "r":
            self._reset()
        elif k == "Return":
            if self._step == "calib" and len(self._calib_pts) >= 4:
                self._go_to_board()
            elif self._step == "board":
                self._go_to_seed()
            elif self._step == "seed":
                self._finish()

    def _reset(self) -> None:
        if self._step == "calib":
            self._calib_pts.clear()
            self._btn_next.config(state=tk.DISABLED)
        elif self._step == "seed":
            self._seed_pts = {"A": [], "B": [], "REF": []}
            self._btn_start.config(state=tk.DISABLED)
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

        src_pts = [(x / self._scale, y / self._scale)
                   for x, y in self._calib_pts]
        # dst_pts bleiben die fixen Feldkoordinaten (Meter) –
        # die Board-Punkte dienen nur zur visuellen Verifikation
        calib = CalibResult(src_pts=src_pts,
                            dst_pts=list(FIELD_COORDS[:len(src_pts)]))

        seed = SeedResult()
        for key, pts in self._seed_pts.items():
            feat_list = (seed.features_a if key == "A"
                         else seed.features_b if key == "B"
                         else seed.features_ref)
            for px, py in pts:
                ox = int(px / self._scale)
                oy = int(py / self._scale)
                fh, fw = self._frame_bgr.shape[:2]
                r = SEED_PATCH
                y1 = max(0, oy - r);  y2 = min(fh, oy + r)
                x1 = max(0, ox - r);  x2 = min(fw, ox + r)
                crop = self._frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                feat = hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)
                feat_list.append(feat)

        return calib, seed
