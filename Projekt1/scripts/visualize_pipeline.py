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

# Projektroot ins sys.path eintragen, damit `src.*` gefunden wird
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
from src.gui.board_renderer import BoardRenderer, BoardState, PlayerSymbol

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
_MODEL_FINETUNED = "finetune/runs/train/weights/best.pt"
MODEL    = sys.argv[5] if len(sys.argv) > 5 else (_MODEL_FINETUNED if Path(_MODEL_FINETUNED).exists() else "yolo11n.pt")
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
                    conf_thresholds={"player": PLAYER_CONF, "goalkeeper": PLAYER_CONF,
                                     "referee": PLAYER_CONF, "ball": 0.25},
                    detect_ball=False)

# Tracker — niedrige Matching-Schwelle für stabilere Tracks
tracker = PlayerTracker(
    frame_rate=int(info.fps),
    track_activation_threshold=0.10,       # sehr niedrig → fast alle Detektionen werden Tracks
    lost_track_buffer=int(info.fps * 5),   # 5 Sekunden Track-Puffer
    minimum_matching_threshold=0.30,       # IoU-Schwelle: sehr tolerant bei schnellen Bewegungen
)

# TeamAssigner – per User-Referenz (team_colors.json) oder K-Means als Fallback
assigner = TeamAssigner()
assigner_fitted = False

_TEAM_COLORS_FILE = Path("team_colors.json")
if _TEAM_COLORS_FILE.exists():
    try:
        _tc = json.loads(_TEAM_COLORS_FILE.read_text())
        _ref_a = np.array(_tc["team_a_hsv"], dtype=np.float32)
        _ref_b = np.array(_tc["team_b_hsv"], dtype=np.float32)
        assigner.fit_from_references(_ref_a, _ref_b)
        assigner_fitted = True
        logger.info("Team-Farben aus '%s' geladen ✓  (manuell kalibriert)", _TEAM_COLORS_FILE)
    except Exception as exc:
        logger.warning("team_colors.json konnte nicht geladen werden: %s – nutze K-Means.", exc)
else:
    logger.info("Kein team_colors.json gefunden – K-Means wird nach %d Frames verwendet.",
                50)

# ---------------------------------------------------------------------------
# Split-View: Video links | Taktikboard rechts
# ---------------------------------------------------------------------------
_BOARD_IMG_PATH = _PROJECT_ROOT / "Taktikboard" / "Taktikboard.png"

# Ausgabeauflösung: beide Panels auf gleiche Höhe skalieren
_OUT_H   = 540
_vid_w   = int(info.width  * _OUT_H / info.height)   # proportional

_board_renderer: BoardRenderer | None = None
_board_w  = 0
_board_bg_bgr: np.ndarray | None = None   # leeres Board als Hintergrund (BGR, target-size)
_board_panel_buf: np.ndarray | None = None  # wiederverwendbarer Puffer

if _BOARD_IMG_PATH.exists():
    # BoardRenderer direkt auf Zielgrösse einrichten (spart Resize-RAM)
    _orig_board = __import__("PIL").Image.open(_BOARD_IMG_PATH)
    _orig_w, _orig_h = _orig_board.size
    _board_w = int(_orig_w * _OUT_H / _orig_h)

    # Hintergrundbild einmalig auf Zielgrösse vorberechnen (BGR)
    _board_bg_bgr = cv2.resize(
        cv2.cvtColor(np.array(_orig_board.convert("RGB")), cv2.COLOR_RGB2BGR),
        (_board_w, _OUT_H), interpolation=cv2.INTER_AREA
    )
    del _orig_board  # Speicher freigeben

    _board_renderer = BoardRenderer(
        board_image_path=_BOARD_IMG_PATH,
        field_width_m=40.0,
        field_height_m=20.0,
    )
    # BoardRenderer-Hintergrundbild auf Zielgrösse skalieren → weniger RAM im Render-Loop
    _bg_small = _board_renderer._bg.resize((_board_w, _OUT_H), __import__("PIL").Image.LANCZOS)
    _board_renderer._bg = _bg_small
    _board_renderer.canvas_w, _board_renderer.canvas_h = _board_w, _OUT_H
    _board_renderer._draw_x0 = int(_board_w * 0.07)
    _board_renderer._draw_y0 = int(_OUT_H  * 0.07)
    _board_renderer._draw_x1 = _board_w - _board_renderer._draw_x0
    _board_renderer._draw_y1 = _OUT_H  - _board_renderer._draw_y0
    _board_renderer._draw_w  = _board_renderer._draw_x1 - _board_renderer._draw_x0
    _board_renderer._draw_h  = _board_renderer._draw_y1 - _board_renderer._draw_y0

    _board_panel_buf = _board_bg_bgr.copy()
    logger.info("Taktikboard geladen ✓  (Render-Auflösung: %dx%d px)", _board_w, _OUT_H)
else:
    logger.warning("Taktikboard.png nicht gefunden – nur Video-Ansicht.")

_OUT_W = _vid_w + _board_w

# Vorallokierter kombinierter Output-Frame (kein np.concatenate pro Frame)
_combined_buf = np.zeros((_OUT_H, _OUT_W, 3), dtype=np.uint8)

# VideoWriter (kombinierte Breite)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT, fourcc, info.fps, (_OUT_W, _OUT_H))
if not writer.isOpened():
    logger.error("VideoWriter konnte nicht geöffnet werden für: %s", OUTPUT)
    sys.exit(1)

logger.info("Ausgabe-Auflösung: %dx%d (Video %dx%d | Board %dx%d)",
            _OUT_W, _OUT_H, _vid_w, _OUT_H, _board_w, _OUT_H)

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
# Erster Durchlauf: Features für TeamAssigner sammeln (nur wenn kein team_colors.json)
# ---------------------------------------------------------------------------
if not assigner_fitted:
    logger.info("Sammle Farbmerkmale für Teamzuordnung (erste 50 Frames) …")
    fit_features = []
    for fi, frame in iter_frames(VIDEO, max_frames=50):
        dets = [d for d in detector.detect(frame, frame_index=fi)
                if d.class_name != "referee"]
        tracked = tracker.update(dets, frame_index=fi)
        used_bboxes = [p.bbox for p in tracked]
        all_player_dets = [d for d in dets if d.class_name in ("player", "goalkeeper")]
        for det in all_player_dets:
            if not any(_iou_xyxy(det.bbox, tb) >= 0.4 for tb in used_bboxes):
                feat = extract_hsv_feature(frame, det.bbox)
                if feat is not None:
                    fit_features.append(feat)
        for player in tracked:
            feat = extract_hsv_feature(frame, player.bbox)
            if feat is not None:
                fit_features.append(feat)

    tracker.reset()

    if len(fit_features) >= 2:
        assigner.fit(fit_features, n_clusters=2)
        assigner_fitted = True
        logger.info("K-Means trainiert auf %d Feature-Vektoren.", len(fit_features))
    else:
        logger.warning("Zu wenig Features für K-Means (%d) – Teams werden nicht eingefärbt.",
                       len(fit_features))
else:
    logger.info("Überspringe Feature-Collection (team_colors.json vorhanden).")

# ---------------------------------------------------------------------------
# Haupt-Durchlauf: annotiertes Video erzeugen
# ---------------------------------------------------------------------------
logger.info("Erzeuge annotiertes Video …")

all_tracked_frames = []
frame_count = 0

# ---------------------------------------------------------------------------
# Stabilitäts-Caches
# ---------------------------------------------------------------------------
from collections import Counter, deque

_SMOOTH_ALPHA    = 0.25   # Bbox-Glättung (kleiner = ruhiger, träger)
_VOTE_WINDOW     = 12     # Frames für Majority-Vote Team-Abstimmung
_BOARD_UPDATE_HZ = 8      # Max. Board-Aktualisierungen pro Sekunde

_bbox_smooth:  dict[int, list[float]]       = {}   # track_id → geglättete Bbox
_team_votes:   dict[int, deque]             = {}   # track_id → letzte N Team-Votes
_team_stable:  dict[int, int]               = {}   # track_id → stabiles Team (Majority)
_last_board_fi: int                         = -999 # letztes Frame mit Board-Update

_board_interval = max(1, int(info.fps / _BOARD_UPDATE_HZ))  # z.B. 3 bei 25fps

# Distanz-Tracker für DET-only: stabile Pseudo-IDs über Frames
# Spieler können max. ~1m pro Frame laufen (bei 30fps, 10m/s Sprint)
# Wir erlauben 4m Toleranz als Puffer für verpasste Frames
_DET_MAX_DIST_M = 4.0        # Meter im Feldkoordinatensystem
_DET_MAX_LOST   = 8          # Frames ohne Sichtung → ID wird verworfen
_det_id_counter: int         = 10_000  # Pseudo-IDs ab 10000 (klar von ByteTrack-IDs getrennt)
_det_tracks: dict[int, dict] = {}
# Format: {det_id: {"bx": float, "by": float, "team": int, "lost": int}}

def _match_det_to_track(bx: float, by: float) -> int:
    """Findet die nächste vorhandene DET-Track-ID oder erstellt eine neue."""
    global _det_id_counter
    best_id, best_dist = None, float("inf")
    for did, info_d in _det_tracks.items():
        dist = ((bx - info_d["bx"]) ** 2 + (by - info_d["by"]) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best_id = dist, did
    if best_id is not None and best_dist <= _DET_MAX_DIST_M:
        return best_id
    # Neue ID vergeben
    _det_id_counter += 1
    return _det_id_counter

def _update_det_tracks(matched: dict[int, tuple[float, float, int]]) -> None:
    """Aktualisiert den DET-Track-State; wirft alte Tracks raus."""
    seen_ids = set(matched.keys())
    # Aktive Tracks aktualisieren
    for did, (bx, by, team) in matched.items():
        _det_tracks[did] = {"bx": bx, "by": by, "team": team, "lost": 0}
    # Nicht-gesehene Tracks altern lassen
    for did in list(_det_tracks.keys()):
        if did not in seen_ids:
            _det_tracks[did]["lost"] += 1
            if _det_tracks[did]["lost"] > _DET_MAX_LOST:
                del _det_tracks[did]


def _vote_team(track_id: int, frame_img, bbox) -> int:
    """Majority-Vote über ein Zeitfenster: verhindert Team-Wechsel durch Einzel-Fehler."""
    if track_id not in _team_votes:
        _team_votes[track_id] = deque(maxlen=_VOTE_WINDOW)

    if assigner_fitted:
        feat = extract_hsv_feature(frame_img, bbox)
        vote = assigner.predict(feat) if feat is not None else -1
    else:
        vote = -1
    _team_votes[track_id].append(vote)

    # Mehrheitsentscheid: häufigstes Team in den letzten N Frames
    votes = list(_team_votes[track_id])
    valid = [v for v in votes if v != -1]
    if valid:
        majority = Counter(valid).most_common(1)[0][0]
    else:
        majority = -1

    # Stabiles Team: erst nach genug Votes setzen, danach beibehalten
    if len(votes) >= _VOTE_WINDOW // 2:
        _team_stable[track_id] = majority
    elif track_id not in _team_stable:
        _team_stable[track_id] = majority
    return _team_stable[track_id]


def _board_update_due(fi: int) -> bool:
    """Rate-Limiter: gibt True zurück wenn das Taktikboard aktualisiert werden darf.

    Ziel: Spielerpositionen nur ~8× pro Sekunde ans Board schicken statt
    bei jedem Frame (25-30×/s). So werden kurze Erkennungsfehler und
    schnelle Team-Wechsel für das menschliche Auge unsichtbar.
    """
    global _last_board_fi
    if fi - _last_board_fi >= _board_interval:
        _last_board_fi = fi
        return True
    return False


def _smooth_bbox(track_id: int, bbox: tuple) -> tuple:
    """Exponential Moving Average auf die Bounding Box anwenden."""
    b = list(bbox)
    if track_id not in _bbox_smooth:
        _bbox_smooth[track_id] = b
    else:
        s = _bbox_smooth[track_id]
        _bbox_smooth[track_id] = [_SMOOTH_ALPHA * b[i] + (1 - _SMOOTH_ALPHA) * s[i]
                                   for i in range(4)]
    return tuple(_bbox_smooth[track_id])

for fi, frame in iter_frames(VIDEO, max_frames=N_FRAMES):

    # Detection + Tracking (Schiedsrichter explizit ausfiltern)
    dets = [d for d in detector.detect(frame, frame_index=fi)
            if d.class_name != "referee"]
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

    # --- Spieler zeichnen + Board-Symbole sammeln ---
    counts = {TEAM_A: 0, TEAM_B: 0, -1: 0}
    _board_symbols: list[PlayerSymbol] = []

    for player in tracked:
        # Geglättete Bbox verwenden (reduziert Zittern)
        smooth_bbox = _smooth_bbox(player.track_id, player.bbox)
        x1, y1, x2, y2 = (int(v) for v in smooth_bbox)

        # Team per Majority-Vote über letzten N Frames (robust gegen Einzel-Fehler)
        team = _vote_team(player.track_id, frame, player.bbox)

        # Feldkoordinaten via Homographie
        bx_m, by_m = 20.0, 10.0  # Feldmitte als Fallback
        if H is not None:
            cx_px = (x1 + x2) / 2.0
            cy_px = float(y2)
            bx_m, by_m = transform_point((cx_px, cy_px), H)

        # Positions-Fallback für Team: Spieler links → Team A, rechts → Team B
        if team == -1 and H is not None:
            team = TEAM_A if bx_m < 20.0 else TEAM_B
        counts[team] = counts.get(team, 0) + 1

        color = TEAM_COLORS[team]

        # Bounding-Box im Video
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label oben: ID + Klasse + Team
        label_top = f"#{player.track_id} {player.class_name[:3].upper()} T{TEAM_LABEL[team]}"
        _put_label(frame, label_top, x1, y1 - 4, color)

        # Label unten: Feldkoordinaten (m)
        if H is not None:
            coord_label = f"{bx_m:.1f},{by_m:.1f}m"
            _put_label(frame, coord_label, x1, y2 + 18, color, scale=0.45, thickness=1)

        # Symbol fürs Board: nur wenn Koordinaten sinnvoll (nicht weit ausserhalb Feld)
        _MARGIN = 5.0  # Meter Toleranz ausserhalb des Felds
        if -_MARGIN <= bx_m <= 40 + _MARGIN and -_MARGIN <= by_m <= 20 + _MARGIN:
            _board_symbols.append(PlayerSymbol(
                track_id=player.track_id,
                team=team,
                class_name=player.class_name,
                board_x=float(np.clip(bx_m, 0, 40)),
                board_y=float(np.clip(by_m, 0, 20)),
                label=str(player.track_id),
            ))

    # Nicht getrackte Detektionen: mit stabilem Distanz-Tracker + Board-Mapping
    tracked_bboxes = [tuple(float(v) for v in p.bbox) for p in tracked]
    det_only_count = 0

    def _center_near(bbox_a, bbox_b) -> bool:
        cx_a = (bbox_a[0] + bbox_a[2]) / 2;  cy_a = (bbox_a[1] + bbox_a[3]) / 2
        cx_b = (bbox_b[0] + bbox_b[2]) / 2;  cy_b = (bbox_b[1] + bbox_b[3]) / 2
        w = max(bbox_a[2]-bbox_a[0], bbox_b[2]-bbox_b[0])
        h = max(bbox_a[3]-bbox_a[1], bbox_b[3]-bbox_b[1])
        return abs(cx_a - cx_b) < w * 0.6 and abs(cy_a - cy_b) < h * 0.6

    _det_matched_this_frame: dict[int, tuple[float, float, int]] = {}

    for det in dets:
        if det.class_name not in ("player", "goalkeeper"):
            continue
        if any(_iou_xyxy(det.bbox, tb) >= 0.20 or _center_near(det.bbox, tb)
               for tb in tracked_bboxes):
            continue
        det_only_count += 1
        x1d, y1d, x2d, y2d = (int(v) for v in det.bbox)

        # Team bestimmen
        if assigner_fitted:
            feat = extract_hsv_feature(frame, det.bbox)
            det_team = assigner.predict(feat) if feat is not None else -1
            box_color = TEAM_COLORS.get(det_team, COLOR_DET_ONLY)
            label = f"PLA T{TEAM_LABEL.get(det_team, '?')}"
        else:
            det_team = -1
            box_color = COLOR_DET_ONLY
            label = f"DET {det.confidence:.2f}"

        cv2.rectangle(frame, (x1d, y1d), (x2d, y2d), box_color, 1, cv2.LINE_AA)
        _put_label(frame, label, x1d, y1d - 4, box_color, scale=0.45, thickness=1)

        # Projektion + Distanz-Tracker
        if H is not None:
            cx_d = (x1d + x2d) / 2.0
            cy_d = float(y2d)
            bx_d, by_d = transform_point((cx_d, cy_d), H)
            if det_team == -1:
                det_team = TEAM_A if bx_d < 20.0 else TEAM_B
            if -_MARGIN <= bx_d <= 40 + _MARGIN and -_MARGIN <= by_d <= 20 + _MARGIN:
                bx_c = float(np.clip(bx_d, 0, 40))
                by_c = float(np.clip(by_d, 0, 20))

                # Stabilen Pseudo-ID per Distanz-Matching finden
                det_id = _match_det_to_track(bx_c, by_c)
                _det_matched_this_frame[det_id] = (bx_c, by_c, det_team)

                # Geglättete Position aus vorherigem Frame verwenden (falls vorhanden)
                if det_id in _det_tracks:
                    prev = _det_tracks[det_id]
                    bx_c = _SMOOTH_ALPHA * bx_c + (1 - _SMOOTH_ALPHA) * prev["bx"]
                    by_c = _SMOOTH_ALPHA * by_c + (1 - _SMOOTH_ALPHA) * prev["by"]

                _board_symbols.append(PlayerSymbol(
                    track_id=det_id,   # stabiler Pseudo-ID (>=10000)
                    team=det_team,
                    class_name=det.class_name,
                    board_x=bx_c,
                    board_y=by_c,
                    label="",          # kein Label → übersichtlicher
                ))

    # DET-Track-State für nächstes Frame aktualisieren
    _update_det_tracks(_det_matched_this_frame)

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

    # --- Taktikboard rendern und Side-by-Side zusammensetzen ---
    # Video-Panel direkt in kombinierten Buffer kopieren (kein extra Alloc)
    cv2.resize(frame, (_vid_w, _OUT_H), dst=_combined_buf[:, :_vid_w], interpolation=cv2.INTER_AREA)

    if _board_renderer and _board_w > 0:
        # Board nur laut Rate-Limiter neu rendern
        if _board_update_due(fi):
            _state = BoardState(players=_board_symbols, frame_index=fi)
            _board_pil = _board_renderer.render_rgb(_state)
            # PIL → BGR direkt in vorallokierten Panel-Puffer
            np.copyto(_board_panel_buf, cv2.cvtColor(np.array(_board_pil), cv2.COLOR_RGB2BGR))
            del _board_pil  # PIL-Bild sofort freigeben

        # Board-Panel in rechte Hälfte des kombinierten Buffers
        _combined_buf[:, _vid_w:] = _board_panel_buf
        # Trennlinie
        _combined_buf[:, _vid_w:_vid_w+2] = (60, 60, 60)
    elif _board_w == 0:
        pass  # nur Video-Panel

    writer.write(_combined_buf)
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
