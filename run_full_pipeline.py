"""Vollständige Pipeline: Annotiertes Video + Taktikboard nebeneinander.

Phase 8 Fixes:
  - Tracking stabilisiert (längerer Buffer, niedrigere Match-Schwelle)
  - ROI-Filter: nur Detektionen innerhalb des Spielfelds
  - Teamfarben: mehr Frames für K-Means + k=3 mit Schiri-Filterung
  - Konfidenz-Tuning für stabile Spieleranzahl
  - Kalibrierung per JSON oder direkt im Script

Aufruf:
    python run_full_pipeline.py [VIDEO] [CALIB_JSON] [N_FRAMES] [OUTPUT]

Beispiel (ganzes Video):
    python run_full_pipeline.py Videos/Trainingsdaten/Mittelland_4.mp4 calibration_mittelland_4.json 0 output_combined.mp4
"""

from __future__ import annotations

import logging
import sys
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

from src.object_detection.detector import Detector
from src.tracking.tracker import PlayerTracker
from src.tracking.team_assigner import TeamAssigner, extract_hsv_feature, TEAM_A, TEAM_B, TEAM_UNKNOWN
from src.video_processing.calibration import load_calibration
from src.video_processing.homography import compute_homography, transform_point
from src.gui.board_renderer import BoardRenderer, BoardState, PlayerSymbol
from src.gui.setup_dialog import SetupDialog

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pipeline")

# ===========================================================================
# KONFIGURATION – hier anpassen
# ===========================================================================

VIDEO      = sys.argv[1] if len(sys.argv) > 1 else "Videos/Mittelland_4.mp4"
CALIB_JSON = sys.argv[2] if len(sys.argv) > 2 else "calibration_mittelland_4.json"
N_FRAMES   = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # 0 = ganzes Video
OUTPUT     = sys.argv[4] if len(sys.argv) > 4 else "output_combined.mp4"
_MODEL_FINETUNED = "finetune/runs/train/weights/best.pt"
MODEL      = _MODEL_FINETUNED if Path(_MODEL_FINETUNED).exists() else "yolo11n.pt"
BOARD_IMG  = "Taktikboard/Taktikboard.png"

# Konfidenz-Schwellen
CONF_PLAYER     = 0.25
CONF_GOALKEEPER = 0.25
CONF_BALL       = 0.30

# Ball-Tracking deaktivieren bis genug Trainingsdaten vorhanden sind
DETECT_BALL = False

# ROI-Margin: Detektionen ausserhalb des Feldes (+ Rand in Metern) werden gefiltert
FIELD_MARGIN_M = 2.0   # 2m Puffer um das Spielfeld

# Anzahl Frames für Team-Fitting
TEAM_FIT_FRAMES = 60

# Taktikboard-Panel Breite
BOARD_OUT_W = 800
# ===========================================================================


# ---------------------------------------------------------------------------
# Hilfsfunktion: ROI-Filter
# ---------------------------------------------------------------------------
def _in_field(cx: float, cy: float, H: np.ndarray,
              margin: float = FIELD_MARGIN_M,
              field_w: float = 40.0, field_h: float = 20.0) -> bool:
    """Gibt True zurück wenn (cx,cy) Pixel innerhalb des Spielfeld-Bereichs liegt."""
    bx, by = transform_point((cx, cy), H)
    return (-margin <= bx <= field_w + margin) and (-margin <= by <= field_h + margin)


# ---------------------------------------------------------------------------
# Setup-Dialog: Kalibrierung + Farb-Seeding
# ---------------------------------------------------------------------------
cap_first = cv2.VideoCapture(VIDEO)
ret, first_frame = cap_first.read()
cap_first.release()
if not ret:
    logger.error("Konnte ersten Frame nicht lesen: %s", VIDEO)
    sys.exit(1)

seed_result = None

# Immer vollständigen Dialog zeigen (Kalibrierung + Seeding)
logger.info("Öffne Setup-Dialog (Kalibrierung + Farb-Seeding) …")
dialog = SetupDialog(first_frame)
calib_result, seed_result = dialog.run()

if calib_result is None:
    logger.error("Kalibrierung abgebrochen.")
    sys.exit(0)

src_pts = calib_result.src_pts
dst_pts = calib_result.dst_pts
H = compute_homography(src_pts, dst_pts)
H_inv = np.linalg.inv(H)


# ---------------------------------------------------------------------------
# Modell, Tracker, TeamAssigner
# ---------------------------------------------------------------------------
detector = Detector(MODEL, conf_thresholds={
    "player":     CONF_PLAYER,
    "goalkeeper": CONF_GOALKEEPER,
    "ball":       CONF_BALL,
})
tracker  = PlayerTracker(
    lost_track_buffer=90,          # 3s bei 30fps – Spieler bleiben länger in Erinnerung
    minimum_matching_threshold=0.7,
)
assigner = TeamAssigner()

# ---------------------------------------------------------------------------
# Video öffnen
# ---------------------------------------------------------------------------
cap          = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_process    = total_frames if N_FRAMES == 0 else min(N_FRAMES, total_frames)
logger.info("Video: %s | %dx%d | %.1f fps | %d/%d Frames", VIDEO, vid_w, vid_h, fps, n_process, total_frames)

# ---------------------------------------------------------------------------
# Taktikboard-Renderer
# ---------------------------------------------------------------------------
renderer    = BoardRenderer(board_image_path=BOARD_IMG)
board_aspect = renderer._bg.height / renderer._bg.width
board_out_h  = int(BOARD_OUT_W * board_aspect)

out_vid_h = 720
out_vid_w = int(vid_w * (out_vid_h / vid_h))
out_w     = out_vid_w + BOARD_OUT_W
out_h     = max(out_vid_h, board_out_h)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (out_w, out_h))
logger.info("Ausgabe: %s | %dx%d", OUTPUT, out_w, out_h)

# ---------------------------------------------------------------------------
# Erster Pass: Team-Fitting mit ROI-Filter (mehr Frames, nur Feldspieler)
# ---------------------------------------------------------------------------
logger.info("Sammle Farbmerkmale für Teamzuordnung (%d Frames, nur Spielfeld) …", TEAM_FIT_FRAMES)
features: list[np.ndarray] = []
pre_cap = cv2.VideoCapture(VIDEO)

for fi in range(min(TEAM_FIT_FRAMES, n_process)):
    ret, frame = pre_cap.read()
    if not ret:
        break
    dets = detector.detect(frame, fi)
    for d in dets:
        if d.class_name not in ("player", "goalkeeper"):
            continue
        cx = (d.bbox[0] + d.bbox[2]) / 2
        cy = (d.bbox[1] + d.bbox[3]) / 2
        if not _in_field(cx, cy, H):
            continue
        feat = extract_hsv_feature(frame, d.bbox)
        if feat is not None:
            features.append(feat)
pre_cap.release()

fit_done = False
# Seed-Features vom Dialog hinzufügen (werden doppelt gewichtet)
if seed_result is not None:
    SEED_WEIGHT = 5  # Jeder manuelle Klick zählt 5× stärker
    for feat in seed_result.features_a:
        features.extend([feat] * SEED_WEIGHT)
    for feat in seed_result.features_b:
        features.extend([feat] * SEED_WEIGHT)
    for feat in seed_result.features_ref:
        features.extend([feat] * SEED_WEIGHT)
    logger.info("Seed-Features: A=%d B=%d Ref=%d",
                len(seed_result.features_a),
                len(seed_result.features_b),
                len(seed_result.features_ref))

n_clusters = 3 if (seed_result and len(seed_result.features_ref) > 0) else 3
if len(features) >= n_clusters:
    assigner.fit(list(features), n_clusters=n_clusters)
    fit_done = True
    logger.info("K-Means trainiert auf %d Feature-Vektoren.", len(features))
else:
    logger.warning("Zu wenige Features (%d) – Teamzuordnung deaktiviert.", len(features))

# Team-Memory mit Auto-Lock + Positions-Backup
team_votes: dict[int, list[int]] = {}
team_locked: dict[int, int] = {}       # track_id → fixiertes Team
track_positions: dict[int, list[float]] = {}  # track_id → Feld-X-Positionen
MAX_PLAYERS = 14
AUTO_LOCK_VOTES = 15
AUTO_LOCK_MIN_RATIO = 0.75

# Prüfe ob Farb-Clustering nützlich ist (Zentren weit genug auseinander)
def _color_clustering_useful() -> bool:
    if not fit_done or assigner._centers is None:
        return False
    team_centers = [assigner._centers[i] for i in range(len(assigner._centers))
                    if assigner._cluster_to_team.get(i, TEAM_UNKNOWN) != TEAM_UNKNOWN]
    if len(team_centers) < 2:
        return False
    dist = np.linalg.norm(team_centers[0] - team_centers[1])
    return float(dist) > 25.0  # Mindestabstand im HSV-Raum

USE_COLOR = _color_clustering_useful()
if fit_done and assigner._centers is not None:
    team_ctrs = [assigner._centers[i] for i in range(len(assigner._centers))
                 if assigner._cluster_to_team.get(i, TEAM_UNKNOWN) != TEAM_UNKNOWN]
    color_dist = float(np.linalg.norm(team_ctrs[0] - team_ctrs[-1])) if len(team_ctrs) >= 2 else 0.0
else:
    color_dist = 0.0
logger.info("Farb-Clustering verwendbar: %s (Zentrenabstand: %.1f)", USE_COLOR, color_dist)

def _get_team(track_id: int, frame: np.ndarray, bbox: tuple,
              field_x: float | None = None) -> int:
    """Team mit Auto-Lock. Nutzt Farbe wenn unterscheidbar, sonst Feldposition."""
    if track_id in team_locked:
        return team_locked[track_id]

    vote = TEAM_UNKNOWN

    # Strategie 1: Farbe (nur wenn Cluster klar getrennt)
    if USE_COLOR and fit_done:
        feat = extract_hsv_feature(frame, bbox)
        if feat is not None:
            vote = assigner.predict(feat)

    # Strategie 2: Feldposition als Fallback (linke Hälfte = A, rechte = B)
    if vote == TEAM_UNKNOWN and field_x is not None:
        # Sammle Positions-Historie für diesen Track
        pos_hist = track_positions.setdefault(track_id, [])
        pos_hist.append(field_x)
        if len(pos_hist) > 60:
            pos_hist.pop(0)
        avg_x = float(np.mean(pos_hist))
        vote = TEAM_A if avg_x < 20.0 else TEAM_B

    if vote == TEAM_UNKNOWN:
        return TEAM_UNKNOWN

    votes = team_votes.setdefault(track_id, [])
    votes.append(vote)
    if len(votes) > 30:
        votes.pop(0)
    majority = max(set(votes), key=votes.count)

    if len(votes) >= AUTO_LOCK_VOTES:
        ratio = votes.count(majority) / len(votes)
        if ratio >= AUTO_LOCK_MIN_RATIO:
            team_locked[track_id] = majority
            logger.info("Auto-Lock: Track #%d → Team %s (%.0f%%)",
                        track_id, "A" if majority == TEAM_A else "B", ratio * 100)
    return majority

# ---------------------------------------------------------------------------
# Hauptschleife
# ---------------------------------------------------------------------------
logger.info("Verarbeite %d Frames …", n_process)
frame_idx = 0

while frame_idx < n_process:
    ret, frame = cap.read()
    if not ret:
        break

    # Detektion + ROI-Filter
    all_dets = detector.detect(frame, frame_idx)
    dets_roi = [
        d for d in all_dets
        if _in_field(
            (d.bbox[0] + d.bbox[2]) / 2,
            (d.bbox[1] + d.bbox[3]) / 2,
            H,
        )
    ]
    # Spieler-Limit: max. MAX_PLAYERS, sortiert nach Konfidenz
    player_dets = sorted(
        [d for d in dets_roi if d.class_name in ("player", "goalkeeper")],
        key=lambda d: d.confidence, reverse=True
    )[:MAX_PLAYERS]
    ball_dets   = [d for d in dets_roi if d.class_name == "ball"] if DETECT_BALL else []
    # referee-Detektionen explizit ignorieren (nicht tracken, nicht anzeigen)
    dets = player_dets + ball_dets

    # Tracking
    tracked = tracker.update(dets, frame_idx)

    # ── Linke Seite: Annotiertes Frame ──────────────────────────────────────
    ann = frame.copy()

    # Spielfeld-Gitter
    for xm in range(0, 41, 5):
        pts = [(float(xm), float(ym)) for ym in np.linspace(0, 20, 30)]
        for pt in pts:
            gx, gy = transform_point(pt, H_inv)
            if 0 <= gx <= vid_w and 0 <= gy <= vid_h:
                cv2.circle(ann, (int(gx), int(gy)), 2, (0, 220, 255), -1)
    for ym in range(0, 21, 5):
        pts = [(float(xm), float(ym)) for xm in np.linspace(0, 40, 50)]
        for pt in pts:
            gx, gy = transform_point(pt, H_inv)
            if 0 <= gx <= vid_w and 0 <= gy <= vid_h:
                cv2.circle(ann, (int(gx), int(gy)), 2, (0, 220, 255), -1)

    # Bounding Boxes + Labels
    players_sym = []
    for tp in tracked:
        cx = (tp.bbox[0] + tp.bbox[2]) / 2
        cy = (tp.bbox[1] + tp.bbox[3]) / 2
        fx, fy = transform_point((cx, cy), H)
        team = _get_team(tp.track_id, frame, tp.bbox, field_x=fx)
        color_bgr = (40, 40, 220) if team == TEAM_A else (220, 90, 30) if team == TEAM_B else (140, 140, 140)
        x1, y1, x2, y2 = (int(v) for v in tp.bbox)
        cv2.rectangle(ann, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"#{tp.track_id} {tp.class_name}"
        cv2.putText(ann, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(ann, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)
        coord_txt = f"{fx:.1f}m,{fy:.1f}m"
        cv2.putText(ann, coord_txt, (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(ann, coord_txt, (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 0), 1, cv2.LINE_AA)
        players_sym.append(PlayerSymbol(
            track_id=tp.track_id, team=team,
            class_name=tp.class_name, board_x=fx, board_y=fy,
        ))

    # Statistik oben links
    n_a = sum(1 for p in players_sym if p.team == TEAM_A)
    n_b = sum(1 for p in players_sym if p.team == TEAM_B)
    stats = f"Frame {frame_idx+1}/{n_process}  |  Team A: {n_a}  Team B: {n_b}  Total: {len(tracked)}"
    cv2.putText(ann, stats, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(ann, stats, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)

    ann_resized = cv2.resize(ann, (out_vid_w, out_vid_h))

    # ── Rechte Seite: Taktikboard ────────────────────────────────────────────
    state      = BoardState(players=players_sym)
    board_pil  = renderer.render(state)
    board_np   = cv2.cvtColor(np.array(board_pil), cv2.COLOR_RGB2BGR)
    board_res  = cv2.resize(board_np, (BOARD_OUT_W, board_out_h))

    # ── Kombinieren ──────────────────────────────────────────────────────────
    combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    combined[:out_vid_h, :out_vid_w]                        = ann_resized
    combined[:board_out_h, out_vid_w:out_vid_w + BOARD_OUT_W] = board_res
    cv2.line(combined, (out_vid_w, 0), (out_vid_w, out_h), (80, 80, 80), 2)

    writer.write(combined)

    if frame_idx % 100 == 0 or frame_idx < 3:
        logger.info("  Frame %d/%d | erkannt: %d (ROI-gefiltert: %d) | A:%d B:%d",
                    frame_idx + 1, n_process, len(tracked),
                    len(all_dets) - len(dets), n_a, n_b)

    frame_idx += 1

cap.release()
writer.release()
logger.info("Fertig! %s", Path(OUTPUT).resolve())
logger.info("Öffnen mit:  xdg-open %s", OUTPUT)
