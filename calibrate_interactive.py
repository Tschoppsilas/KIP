"""Interaktives Kalibrierungs-Tool fuer Phase 2 (OpenCV-Fenster).

Ablauf:
  1. Erster Frame aus dem Video wird in einem Fenster angezeigt
  2. Du klickst nacheinander auf 6 Spielfeldpunkte
  3. Homography wird berechnet
  4. Ergebnis wird als PNG gespeichert + Kalibrierung als JSON

Aufruf:
    ./.venv/bin/python calibrate_interactive.py Videos/Muenchenstein_1.mp4
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

# Qt5 auf XCB (X11/XWayland) zwingen.
# In dieser Umgebung ist laut Runtime nur das xcb-Plugin verfuegbar.
os.environ["QT_QPA_PLATFORM"] = "xcb"

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from video_processing.video_reader import read_first_frame, get_video_info
from video_processing.homography import compute_homography, transform_points
from video_processing.calibration import save_calibration

VIDEO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Videos/Muenchenstein_1.mp4")
DEBUG_LOG_PATH = Path("/home/admin/KIP/.cursor/debug-4e2d6a.log")
DEBUG_SESSION_ID = "4e2d6a"
DEBUG_RUN_ID = f"run_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": DEBUG_RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        # Debug logging must never break calibration flow.
        pass


# region agent log
_debug_log(
    "H1",
    "calibrate_interactive.py:start",
    "script_start_env",
    {
        "argv_len": len(sys.argv),
        "video_path": str(VIDEO),
        "display": os.environ.get("DISPLAY", ""),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY", ""),
        "qt_qpa_platform": os.environ.get("QT_QPA_PLATFORM", ""),
    },
)
# endregion

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
COLORS_BGR = [
    (0,   80, 255),
    (0,  200, 255),
    (0,  255, 100),
    (0,  255,   0),
    (255, 180,  0),
    (255,  80,  0),
]
N_POINTS = len(FIELD_NAMES)

# ─── Video laden ─────────────────────────────────────────────────────────────
print(f"\nLade Video: {VIDEO}")
info  = get_video_info(VIDEO)
frame = read_first_frame(VIDEO)
print(f"Aufloesung: {info.width}x{info.height} | {info.fps} fps | {info.frame_count} Frames")
# region agent log
_debug_log(
    "H2",
    "calibrate_interactive.py:video_loaded",
    "video_loaded_first_frame_ok",
    {
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "frame_count": info.frame_count,
        "frame_shape": list(frame.shape),
    },
)
# endregion

# Fuer Display auf max. 1440 px Breite skalieren
MAX_W = 1440
scale = min(1.0, MAX_W / info.width)
disp_w = int(info.width  * scale)
disp_h = int(info.height * scale)
display = cv2.resize(frame, (disp_w, disp_h))

clicked: list[tuple[int, int]] = []

def _draw(img: np.ndarray) -> np.ndarray:
    canvas = img.copy()
    next_idx = len(clicked)

    # Naechster erwarteter Punkt: Hinweis einblenden
    if next_idx < N_POINTS:
        name  = FIELD_NAMES[next_idx]
        color = COLORS_BGR[next_idx]
        cv2.putText(canvas, f"Klicke Punkt {next_idx+1}/{N_POINTS}: {name}",
                    (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(canvas, f"Klicke Punkt {next_idx+1}/{N_POINTS}: {name}",
                    (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,   2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "Alle Punkte gesetzt – Enter zum Bestaetigen | R = Reset | ESC = Abbruch",
                    (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(canvas, "Alle Punkte gesetzt – Enter zum Bestaetigen | R = Reset | ESC = Abbruch",
                    (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,100), 2, cv2.LINE_AA)

    # Bereits gesetzte Punkte
    for i, (px, py) in enumerate(clicked):
        c = COLORS_BGR[i]
        cv2.circle(canvas, (px, py), 9, (0,0,0), -1)
        cv2.circle(canvas, (px, py), 7, c, -1)
        label = f"{i+1} {FIELD_NAMES[i]}"
        cv2.putText(canvas, label, (px+12, py+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (px+12, py+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c,     1, cv2.LINE_AA)

    return canvas

def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < N_POINTS:
        clicked.append((x, y))
        # region agent log
        _debug_log(
            "H4",
            "calibrate_interactive.py:on_mouse",
            "mouse_click_registered",
            {"x": x, "y": y, "clicked_count": len(clicked)},
        )
        # endregion
        print(f"  Punkt {len(clicked)}: ({x}, {y})  →  {FIELD_NAMES[len(clicked)-1]}")

WINDOW = "UniVision2Board-Calibration (Enter=OK | R=Reset | ESC=Abort)"
# region agent log
_debug_log(
    "H6",
    "calibrate_interactive.py:window_init",
    "window_title_selected",
    {"window_title": WINDOW, "is_ascii": WINDOW.isascii()},
)
# endregion
try:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, disp_w, disp_h)

    callback_set = False
    for attempt in range(1, 4):
        cv2.imshow(WINDOW, _draw(display))
        cv2.waitKey(1)  # Event-Pump, damit Qt das Fenster wirklich initialisiert
        visible = float(cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE))
        # region agent log
        _debug_log(
            "H3",
            "calibrate_interactive.py:window_init",
            "window_probe",
            {"attempt": attempt, "visible": visible},
        )
        # endregion
        if visible < 1.0:
            continue
        try:
            cv2.setMouseCallback(WINDOW, _on_mouse)
            callback_set = True
            # region agent log
            _debug_log(
                "H3",
                "calibrate_interactive.py:window_init",
                "mouse_callback_set",
                {"attempt": attempt, "window": WINDOW, "disp_w": disp_w, "disp_h": disp_h},
            )
            # endregion
            break
        except cv2.error as callback_exc:
            # region agent log
            _debug_log(
                "H3",
                "calibrate_interactive.py:window_init",
                "mouse_callback_failed_attempt",
                {"attempt": attempt, "error": str(callback_exc)},
            )
            # endregion

    if not callback_set:
        raise RuntimeError(
            "Kalibrierungsfenster konnte nicht stabil initialisiert werden "
            "(OpenCV Mouse-Callback). Bitte pruefe Display/Wayland-Konfiguration."
        )

    # region agent log
    _debug_log(
        "H3",
        "calibrate_interactive.py:window_init",
        "window_initialized",
        {"window": WINDOW, "disp_w": disp_w, "disp_h": disp_h},
    )
    # endregion
except cv2.error as exc:
    # region agent log
    _debug_log(
        "H3",
        "calibrate_interactive.py:window_init",
        "window_init_failed",
        {"error": str(exc)},
    )
    # endregion
    raise

print(f"\nFenster geoeffnet – klicke {N_POINTS} Punkte in dieser Reihenfolge:")
for i, n in enumerate(FIELD_NAMES):
    print(f"  {i+1}. {n}  →  {FIELD_COORDS[i]}")
print("\nEnter = bestaetigen (ab 4 Punkten) | R = zuruecksetzen | ESC = abbrechen\n")

result_pts = None
while True:
    cv2.imshow(WINDOW, _draw(display))
    key = cv2.waitKey(20) & 0xFF
    if key not in (255,):
        # region agent log
        _debug_log(
            "H5",
            "calibrate_interactive.py:event_loop",
            "key_event_detected",
            {"key": int(key), "clicked_count": len(clicked)},
        )
        # endregion
    if key == 27:                              # ESC
        print("Abgebrochen.")
        break
    elif key == ord("r"):                      # R = Reset
        clicked.clear()
        print("Punkte zurueckgesetzt.")
    elif key in (13, 10) and len(clicked) >= 4:  # Enter
        result_pts = list(clicked)
        print(f"\n{len(result_pts)} Punkte bestaetigt.")
        break

cv2.destroyAllWindows()

if not result_pts:
    sys.exit(0)

# ─── Homography berechnen ─────────────────────────────────────────────────────
n = len(result_pts)
src_px  = [(x / scale, y / scale) for x, y in result_pts]   # Zurueck auf Originalaufloesung
dst_m   = list(FIELD_COORDS[:n])

H = compute_homography(src_px, dst_m)
print(f"\nHomography-Matrix:\n{H}\n")
# region agent log
_debug_log(
    "H5",
    "calibrate_interactive.py:homography",
    "homography_computed",
    {"points_used": n, "matrix_shape": list(H.shape)},
)
# endregion

# Test: Spielfeldmitte
from video_processing.homography import transform_point
cx, cy = transform_point((info.width/2, info.height/2), H)
print(f"Bildmitte ({info.width//2}, {info.height//2}) → Board ({cx:.2f} m, {cy:.2f} m)")

# ─── Ergebnis-Bild speichern ─────────────────────────────────────────────────
result_frame = frame.copy()
for i, (px, py) in enumerate(src_px):
    c = COLORS_BGR[i]
    cv2.circle(result_frame, (int(px), int(py)), 14, (0,0,0), -1)
    cv2.circle(result_frame, (int(px), int(py)), 11, c, -1)
    label = f"{i+1}: {FIELD_NAMES[i]} ({dst_m[i][0]:.0f}m,{dst_m[i][1]:.0f}m)"
    cv2.putText(result_frame, label, (int(px)+16, int(py)+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(result_frame, label, (int(px)+16, int(py)+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2, cv2.LINE_AA)

# Spielfeld-Gitter projizieren
H_inv = np.linalg.inv(H)
for x in range(0, 41, 5):
    line_pts = [(float(x), float(y)) for y in np.linspace(0, 20, 30)]
    px_pts = transform_points(line_pts, H_inv)
    for (gx, gy) in px_pts:
        if 0 <= gx <= info.width and 0 <= gy <= info.height:
            cv2.circle(result_frame, (int(gx), int(gy)), 2, (0, 220, 255), -1)
for y in range(0, 21, 5):
    line_pts = [(float(x), float(y)) for x in np.linspace(0, 40, 50)]
    px_pts = transform_points(line_pts, H_inv)
    for (gx, gy) in px_pts:
        if 0 <= gx <= info.width and 0 <= gy <= info.height:
            cv2.circle(result_frame, (int(gx), int(gy)), 2, (0, 220, 255), -1)

out_png = Path("calibration_result.png")
cv2.imwrite(str(out_png), result_frame)
print(f"Ergebnis-Bild gespeichert: {out_png.resolve()}")

# ─── Kalibrierung als JSON speichern ─────────────────────────────────────────
calib_path = Path("calibration_muenchenstein1.json")
save_calibration(src_px, dst_m, calib_path,
                 metadata={"video": VIDEO.name, "field_width_m": 40, "field_height_m": 20})
print(f"Kalibrierung gespeichert:  {calib_path.resolve()}")
print("\nFertig! Oeffne calibration_result.png um das Ergebnis zu sehen.")
