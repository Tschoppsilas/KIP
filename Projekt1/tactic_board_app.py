"""Taktikboard-Launcher: Pipeline Phase 1–5 + interaktive GUI (Phase 6).

Führt die komplette Verarbeitungskette durch und öffnet anschliessend
das interaktive Taktikboard in einem Tkinter-Fenster.

Aufruf (aus dem Projektroot, venv aktiv, Desktop-Terminal empfohlen):
    python tactic_board_app.py [VIDEO] [N_FRAMES] [CALIB_JSON]

Beispiele:
    python tactic_board_app.py
    python tactic_board_app.py Videos/Muenchenstein_1.mp4 120
    python tactic_board_app.py Videos/Muenchenstein_2.mp4 60 calibration_m2.json
"""

from __future__ import annotations

import os
import sys
import logging

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from src.utils.logging_config import configure_logging
configure_logging(level=logging.INFO)
logger = logging.getLogger("tactic_board_app")

# ---------------------------------------------------------------------------
# Argumente
# ---------------------------------------------------------------------------
VIDEO      = sys.argv[1] if len(sys.argv) > 1 else "Videos/Muenchenstein_1.mp4"
N_FRAMES   = int(sys.argv[2]) if len(sys.argv) > 2 else 90
CALIB_JSON = sys.argv[3] if len(sys.argv) > 3 else "calibration_muenchenstein1.json"
_MODEL_FINETUNED = "finetune/runs/train/weights/best.pt"
MODEL      = _MODEL_FINETUNED if __import__("pathlib").Path(_MODEL_FINETUNED).exists() else "yolo11n.pt"
BOARD_IMG  = "Taktikboard/Taktikboard.png"

logger.info("=== Taktikboard-App ===")
logger.info("Video: %s | Frames: %d", VIDEO, N_FRAMES)

# ---------------------------------------------------------------------------
# Phase 2 – Kalibrierung + Homographie
# ---------------------------------------------------------------------------
from src.video_processing.video_reader import get_video_info, iter_frames
from src.video_processing.calibration import load_calibration
from src.video_processing.homography import compute_homography, transform_point

info = get_video_info(VIDEO)
logger.info("Auflösung: %dx%d | %.1f fps", info.width, info.height, info.fps)

H = None
field_w, field_h = 40.0, 20.0
try:
    src_pts, dst_pts = load_calibration(CALIB_JSON)
    H = compute_homography(src_pts, dst_pts)
    xs = [p[0] for p in dst_pts]
    ys = [p[1] for p in dst_pts]
    field_w = max(xs) - min(xs)
    field_h = max(ys) - min(ys)
    logger.info("Kalibrierung geladen ✓  (Feld %.0f×%.0f m)", field_w, field_h)
except FileNotFoundError:
    logger.warning("Keine Kalibrierungsdatei – Feldkoordinaten auf Schätzwerte gesetzt.")

# ---------------------------------------------------------------------------
# Phase 3 – YOLO-Detection
# ---------------------------------------------------------------------------
from src.object_detection.detector import Detector

logger.info("Lade YOLO-Modell …")
detector = Detector(MODEL, conf_thresholds={"player": 0.35, "goalkeeper": 0.35, "ball": 0.40},
                    detect_ball=True)

# ---------------------------------------------------------------------------
# Phase 4 – ByteTrack
# ---------------------------------------------------------------------------
from src.tracking.tracker import PlayerTracker
from src.tracking.track import build_trajectories

tracker = PlayerTracker(frame_rate=int(info.fps))

# ---------------------------------------------------------------------------
# Phase 5 – TeamAssigner vorbereiten (auf ersten 30 Frames)
# ---------------------------------------------------------------------------
from src.tracking.team_assigner import TeamAssigner, extract_hsv_feature, TEAM_A, TEAM_B

assigner = TeamAssigner()
assigner_fitted = False
fit_features = []

logger.info("Sammle Farbmerkmale (erste 30 Frames) …")
for fi, frame in iter_frames(VIDEO, max_frames=30):
    dets = detector.detect(frame, frame_index=fi)
    tracked = tracker.update(dets, frame_index=fi)
    for p in tracked:
        feat = extract_hsv_feature(frame, p.bbox)
        if feat is not None:
            fit_features.append(feat)
    if len(fit_features) >= 30:
        break

tracker.reset()

if len(fit_features) >= 2:
    assigner.fit(fit_features)
    assigner_fitted = True
    logger.info("K-Means trainiert auf %d Vektoren.", len(fit_features))

# ---------------------------------------------------------------------------
# Haupt-Durchlauf: BoardState pro Frame aufbauen
# ---------------------------------------------------------------------------
from src.gui.board_renderer import BoardRenderer, BoardState, PlayerSymbol, Arrow

renderer = BoardRenderer(BOARD_IMG, field_width_m=field_w, field_height_m=field_h)
board_states: list[BoardState] = []
all_tracked = []

logger.info("Verarbeite %d Frames …", N_FRAMES)

for fi, frame in iter_frames(VIDEO, max_frames=N_FRAMES):
    dets = detector.detect(frame, frame_index=fi)
    tracked = tracker.update(dets, frame_index=fi)
    all_tracked.append(tracked)

    symbols: list[PlayerSymbol] = []
    for p in tracked:
        # Fußpunkt des Spielers → Feldkoordinaten via Homographie
        cx = (p.bbox[0] + p.bbox[2]) / 2.0
        foot_y = p.bbox[3]
        if H is not None:
            bx, by = transform_point((cx, foot_y), H)
        else:
            # Fallback: Pixel-Position linear skaliert
            bx = cx / info.width * field_w
            by = foot_y / info.height * field_h

        # Team bestimmen
        feat = extract_hsv_feature(frame, p.bbox)
        team = assigner.get_team(p.track_id, feat) if (assigner_fitted and feat is not None) else -1

        symbols.append(PlayerSymbol(
            track_id=p.track_id,
            team=team,
            class_name=p.class_name,
            board_x=max(0.0, min(field_w, bx)),
            board_y=max(0.0, min(field_h, by)),
        ))

    board_states.append(BoardState(players=symbols, frame_index=fi))

# ---------------------------------------------------------------------------
# Regelbasierte Pass-Vorschläge (Phase 6 Must: Passvorschläge als Pfeile)
# ---------------------------------------------------------------------------

def _add_pass_suggestions(states: list[BoardState], max_dist_m: float = 8.0) -> None:
    """Fügt automatische Pass-Vorschläge (grüne Pfeile) zwischen nah beieinander
    stehenden Spielern desselben Teams hinzu.  Nur im letzten Frame sichtbar."""
    if not states:
        return
    last = states[-1]
    added: set[tuple[int, int]] = set()
    players = [p for p in last.players if p.class_name != "ball"]
    for i, a in enumerate(players):
        for j, b in enumerate(players):
            if i >= j or a.team != b.team or a.team == -1:
                continue
            import math
            dist = math.hypot(a.board_x - b.board_x, a.board_y - b.board_y)
            if dist <= max_dist_m and (i, j) not in added:
                last.arrows.append(Arrow(a.board_x, a.board_y, b.board_x, b.board_y, kind="pass"))
                added.add((i, j))

_add_pass_suggestions(board_states)

trajectories = build_trajectories(all_tracked)
logger.info("Fertig: %d Spieler verfolgt, %d Board-States.", len(trajectories), len(board_states))

# ---------------------------------------------------------------------------
# GUI öffnen
# ---------------------------------------------------------------------------
from src.gui.tactic_board import TacticBoardApp

def _on_override(track_id: int, new_team: int) -> None:
    assigner.override(track_id, new_team)
    logger.info("Team-Korrektur: Spieler #%d → Team %s",
                track_id, "A" if new_team == TEAM_A else "B")

logger.info("Öffne Taktikboard-GUI …")
logger.info("(Hotkeys: P=Pass, S=Schuss, R=Laufweg, ESC=Auswahl, ◀▶=Frames)")

app = TacticBoardApp(board_states, renderer, on_team_override=_on_override)
app.run()
