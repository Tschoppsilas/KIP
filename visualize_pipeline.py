"""Vollständige visuelle Pipeline: Kalibrierung → Detection → Tracking → Teamzuordnung.

Erzeugt ein annotiertes Ausgabe-Video (output_annotated.mp4), das du in jedem
Videoplayer öffnen kannst.  Kein Fenster nötig — läuft auch ohne Desktop.

Annotierungen im Video:
  - Kalibrierungspunkte (gelbe Kreise + Labels)
  - Kalibriertes Spielfeld-Raster (grüne Linien)
  - Bounding-Boxes pro Spieler (rot = Team A, blau = Team B)
  - Track-ID + Klasse als Label über der Box
  - Feldkoordinaten (Meter) unter der Box, wenn Homographie vorhanden
  - Statistik-Overlay oben links (Frame-Nr., Detektionen, Spieler je Team)

Aufruf:
    python visualize_pipeline.py [VIDEO] [N_FRAMES] [OUTPUT] [PLAYER_CONF] [MODEL]

Beispiele:
    python visualize_pipeline.py                                   # Defaults
    python visualize_pipeline.py Videos/Muenchenstein_1.mp4 120   # 120 Frames
    python visualize_pipeline.py Videos/Muenchenstein_2.mp4 60 out.mp4
    python visualize_pipeline.py Videos/Muenchenstein_1.mp4 120 out.mp4 0.18
    python visualize_pipeline.py Videos/Muenchenstein_1.mp4 120 out.mp4 0.20 yolo11s.pt
"""

from __future__ import annotations

import json
import os
import sys
import logging
import time
import uuid
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

from src.utils.logging_config import configure_logging
from src.video_processing.video_reader import read_first_frame, get_video_info, iter_frames
from src.video_processing.calibration import load_calibration
from src.video_processing.homography import compute_homography, transform_point
from src.object_detection.detector import Detector
from src.tracking.tracker import PlayerTracker
from src.tracking.track import build_trajectories
from src.tracking.team_assigner import TeamAssigner, extract_hsv_feature, TEAM_A, TEAM_B

configure_logging(level=logging.INFO)
logger = logging.getLogger("visualize")
DEBUG_LOG_PATH = Path("/home/admin/KIP/.cursor/debug-4e2d6a.log")
DEBUG_SESSION_ID = "4e2d6a"
DEBUG_RUN_ID = f"viz_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": DEBUG_RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Argumente
# ---------------------------------------------------------------------------
VIDEO    = sys.argv[1] if len(sys.argv) > 1 else "Videos/Muenchenstein_1.mp4"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 90
OUTPUT   = sys.argv[3] if len(sys.argv) > 3 else "output_annotated.mp4"
PLAYER_CONF = float(sys.argv[4]) if len(sys.argv) > 4 else 0.20
MODEL    = sys.argv[5] if len(sys.argv) > 5 else "finetune/runs/train/weights/best.pt"
CALIB    = "calibration_muenchenstein1.json"

# ---------------------------------------------------------------------------
# Farben (BGR)
# ---------------------------------------------------------------------------
COLOR_TEAM_A   = (0,   60, 220)   # Rot
COLOR_TEAM_B   = (220, 60,   0)   # Blau
COLOR_UNKNOWN  = (160, 160, 160)  # Grau
COLOR_CALIB_PT = (0,  220, 220)   # Gelb
COLOR_GRID     = (0,  180,   0)   # Grün
COLOR_TEXT_BG  = (20,  20,  20)
COLOR_DET_ONLY = (0, 255, 255)    # Cyan/Gelb: erkannt, aber nicht getrackt

TEAM_COLORS = {TEAM_A: COLOR_TEAM_A, TEAM_B: COLOR_TEAM_B, -1: COLOR_UNKNOWN}
TEAM_LABEL  = {TEAM_A: "A", TEAM_B: "B", -1: "?"}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logger.info("Video: %s | Frames: %d | Output: %s", VIDEO, N_FRAMES, OUTPUT)
# region agent log
_debug_log(
    "H2",
    "visualize_pipeline.py:startup",
    "pipeline_start_config",
    {
        "video": VIDEO,
        "n_frames": N_FRAMES,
        "output": OUTPUT,
        "model": MODEL,
        "player_conf": PLAYER_CONF,
    },
)
# endregion

info = get_video_info(VIDEO)
logger.info("Auflösung: %dx%d | %.1f fps", info.width, info.height, info.fps)

# Homographie
H = None
src_pts_calib, dst_pts_calib = [], []
try:
    src_pts_calib, dst_pts_calib = load_calibration(CALIB)
    H = compute_homography(src_pts_calib, dst_pts_calib)
    logger.info("Kalibrierung geladen ✓  (%d Punkte)", len(src_pts_calib))
except FileNotFoundError:
    logger.warning("Keine Kalibrierungsdatei — Feldkoordinaten werden nicht angezeigt.")

# Detector
logger.info("Lade YOLO-Modell …")
detector = Detector(
                    MODEL,
                    conf_thresholds={"player": PLAYER_CONF, "goalkeeper": PLAYER_CONF, "ball": 0.25},
                    detect_ball=True)

# Tracker
tracker = PlayerTracker(frame_rate=int(info.fps))

# TeamAssigner – wird nach den ersten 30 Frames initialisiert
assigner = TeamAssigner()
assigner_fitted = False

# VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT, fourcc, info.fps, (info.width, info.height))
if not writer.isOpened():
    logger.error("VideoWriter konnte nicht geöffnet werden für: %s", OUTPUT)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Feldgitter vorberechnen (für Kalibrierungsoverlay)
# ---------------------------------------------------------------------------
def _field_grid_overlay(frame: np.ndarray,
                         src_pts: list, dst_pts: list, H_mat: np.ndarray) -> np.ndarray:
    """Zeichnet ein 5×5-Raster des kalibrierten Feldes ins Frame."""
    if H_mat is None or not dst_pts:
        return frame

    # Feldgrösse aus dst_pts ableiten
    xs = [p[0] for p in dst_pts]
    ys = [p[1] for p in dst_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Inverse Homographie: Board → Pixel
    H_inv = np.linalg.inv(H_mat)

    def board_to_px(bx: float, by: float) -> tuple[int, int] | None:
        pt = np.array([[[bx, by]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, H_inv)
        x, y = res[0][0]
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            return int(x), int(y)
        return None

    steps_x = 5
    steps_y = 4
    dx = (x_max - x_min) / steps_x
    dy = (y_max - y_min) / steps_y

    # Horizontale Linien
    for i in range(steps_y + 1):
        by = y_min + i * dy
        pts_row = [board_to_px(x_min + j * dx / 10, by) for j in range(steps_x * 10 + 1)]
        pts_row = [p for p in pts_row if p is not None]
        for k in range(len(pts_row) - 1):
            cv2.line(frame, pts_row[k], pts_row[k + 1], COLOR_GRID, 1, cv2.LINE_AA)

    # Vertikale Linien
    for i in range(steps_x + 1):
        bx = x_min + i * dx
        pts_col = [board_to_px(bx, y_min + j * dy / 10) for j in range(steps_y * 10 + 1)]
        pts_col = [p for p in pts_col if p is not None]
        for k in range(len(pts_col) - 1):
            cv2.line(frame, pts_col[k], pts_col[k + 1], COLOR_GRID, 1, cv2.LINE_AA)

    return frame


def _draw_calib_points(frame: np.ndarray, src_pts: list) -> np.ndarray:
    """Markiert die Kalibrierungspunkte im Frame."""
    for i, (px, py) in enumerate(src_pts):
        cx, cy = int(px), int(py)
        cv2.circle(frame, (cx, cy), 10, COLOR_CALIB_PT, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy),  3, COLOR_CALIB_PT, -1)
        cv2.putText(frame, f"K{i+1}", (cx + 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_CALIB_PT, 2, cv2.LINE_AA)
    return frame


def _put_label(frame: np.ndarray, text: str, x: int, y: int,
               color: tuple, scale: float = 0.55, thickness: int = 2) -> None:
    """Schreibt Text mit dunklem Hintergrund für Lesbarkeit."""
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(frame, (x - 2, y - th - bl - 2), (x + tw + 2, y + 2),
                  COLOR_TEXT_BG, -1)
    cv2.putText(frame, text, (x, y - bl),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Erster Durchlauf: Features für TeamAssigner sammeln (stille Phase)
# ---------------------------------------------------------------------------
logger.info("Sammle Farbmerkmale für Teamzuordnung (erste 30 Frames) …")
fit_features = []
fit_frames_used = 0
for fi, frame in iter_frames(VIDEO, max_frames=30):
    dets = detector.detect(frame, frame_index=fi)
    tracked = tracker.update(dets, frame_index=fi)
    for player in tracked:
        feat = extract_hsv_feature(frame, player.bbox)
        if feat is not None:
            fit_features.append(feat)
    fit_frames_used = fi + 1

tracker.reset()   # Tracker für den Haupt-Durchlauf zurücksetzen

if len(fit_features) >= 2:
    assigner.fit(fit_features)
    assigner_fitted = True
    logger.info("K-Means trainiert auf %d Feature-Vektoren.", len(fit_features))
else:
    logger.warning("Zu wenig Features für K-Means (%d) – Teams werden nicht eingefärbt.",
                   len(fit_features))

# ---------------------------------------------------------------------------
# Haupt-Durchlauf: annotiertes Video erzeugen
# ---------------------------------------------------------------------------
logger.info("Erzeuge annotiertes Video …")

all_tracked_frames = []
frame_count = 0

for fi, frame in iter_frames(VIDEO, max_frames=N_FRAMES):

    # Detection + Tracking
    dets = detector.detect(frame, frame_index=fi)
    tracked = tracker.update(dets, frame_index=fi)
    all_tracked_frames.append(tracked)
    if fi < 5 or fi % 30 == 0:
        # region agent log
        _debug_log(
            "H3",
            "visualize_pipeline.py:main_loop",
            "frame_detection_tracking_counts",
            {
                "frame_index": fi,
                "detections_count": len(dets),
                "tracked_count": len(tracked),
                "tracking_drop_count": max(0, len(dets) - len(tracked)),
            },
        )
        # endregion

    # --- Kalibrierungsoverlay ---
    if H is not None:
        frame = _field_grid_overlay(frame, src_pts_calib, dst_pts_calib, H)
        frame = _draw_calib_points(frame, src_pts_calib)

    # --- Spieler zeichnen ---
    counts = {TEAM_A: 0, TEAM_B: 0, -1: 0}
    for player in tracked:
        x1, y1, x2, y2 = (int(v) for v in player.bbox)

        # Team bestimmen
        if assigner_fitted:
            feat = extract_hsv_feature(frame, player.bbox)
            team = assigner.get_team(player.track_id, feat) if feat is not None else -1
        else:
            team = -1
        counts[team] = counts.get(team, 0) + 1

        color = TEAM_COLORS[team]

        # Bounding-Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label oben: ID + Klasse + Team
        label_top = f"#{player.track_id} {player.class_name[:3].upper()} T{TEAM_LABEL[team]}"
        _put_label(frame, label_top, x1, y1 - 4, color)

        # Label unten: Feldkoordinaten (m)
        if H is not None:
            cx = (x1 + x2) / 2.0
            cy = float(y2)
            bx, by = transform_point((cx, cy), H)
            coord_label = f"{bx:.1f},{by:.1f}m"
            _put_label(frame, coord_label, x1, y2 + 18, color, scale=0.45, thickness=1)

    # Nicht getrackte, aber erkannte Spieler trotzdem sichtbar machen.
    tracked_bboxes = [tuple(float(v) for v in p.bbox) for p in tracked]
    det_only_count = 0
    for det in dets:
        if det.class_name not in ("player", "goalkeeper"):
            continue
        if any(_iou_xyxy(det.bbox, tb) >= 0.5 for tb in tracked_bboxes):
            continue
        det_only_count += 1
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DET_ONLY, 1, cv2.LINE_AA)
        _put_label(frame, f"DET {det.confidence:.2f}", x1, y1 - 4, COLOR_DET_ONLY, scale=0.45, thickness=1)

    if fi < 5 or fi % 30 == 0:
        # region agent log
        _debug_log(
            "H5",
            "visualize_pipeline.py:main_loop",
            "det_only_overlay_count",
            {
                "frame_index": fi,
                "det_only_count": det_only_count,
                "detections_count": len(dets),
                "tracked_count": len(tracked),
            },
        )
        # endregion

    # --- Statistik-Overlay ---
    stat_lines = [
        f"Frame {fi+1:4d}/{N_FRAMES}",
        f"Det:  {len(dets):2d}",
        f"Det-only:      {det_only_count:2d}",
        f"Team A (rot):  {counts[TEAM_A]:2d}",
        f"Team B (blau): {counts[TEAM_B]:2d}",
    ]
    for li, line in enumerate(stat_lines):
        y_pos = 28 + li * 26
        _put_label(frame, line, 12, y_pos,
                   color=(220, 220, 220), scale=0.6, thickness=1)

    writer.write(frame)
    frame_count += 1
    if frame_count % 30 == 0:
        logger.info("  … Frame %d/%d", frame_count, N_FRAMES)

writer.release()

# ---------------------------------------------------------------------------
# Trajektorien-Zusammenfassung
# ---------------------------------------------------------------------------
trajectories = build_trajectories(all_tracked_frames)
logger.info("Fertig!")
logger.info("  Ausgabe-Video:   %s", Path(OUTPUT).resolve())
logger.info("  Frames:          %d", frame_count)
logger.info("  Spieler (IDs):   %d", len(trajectories))
logger.info("  Detektionen:     %d", sum(len(f) for f in all_tracked_frames))
logger.info("")
logger.info("Öffne das Video mit:")
logger.info("  xdg-open %s", OUTPUT)
# region agent log
_debug_log(
    "H4",
    "visualize_pipeline.py:summary",
    "pipeline_summary_counts",
    {
        "frames_written": frame_count,
        "total_tracked_instances": sum(len(f) for f in all_tracked_frames),
        "unique_track_ids": len(trajectories),
    },
)
# endregion
