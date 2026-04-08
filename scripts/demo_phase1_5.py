"""Demo-Script: UniVision2Board Phase 1–5.

Zeigt den kompletten Verarbeitungsstand:
  Phase 1 – Logging + Video-Frame laden
  Phase 2 – Video-Info, Homographie, Kalibrierung laden
  Phase 3 – YOLO-Detection (erste N Frames)
  Phase 4 – ByteTrack-Tracking
  Phase 5 – Automatische Teamzuordnung + Beispiel-Korrektur

Aufruf (aus dem Projektroot, venv aktiv):
    python demo_phase1_5.py

Optional: anderes Video über Kommandozeile:
    python demo_phase1_5.py Videos/Muenchenstein_2.mp4
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Qt5/XCB auf Wayland-Desktops erzwingen
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# ---------------------------------------------------------------------------
# Phase 1 – Logging
# ---------------------------------------------------------------------------
from src.utils.logging_config import configure_logging
configure_logging(level=logging.INFO)
logger = logging.getLogger("demo")

logger.info("=== UniVision2Board Demo – Phase 1–5 ===")

# ---------------------------------------------------------------------------
# Video-Pfad
# ---------------------------------------------------------------------------
VIDEO = sys.argv[1] if len(sys.argv) > 1 else "Videos/Muenchenstein_1.mp4"
CALIB_JSON = "calibration_muenchenstein1.json"
N_FRAMES = 30   # Anzahl Frames für Detection + Tracking

# ---------------------------------------------------------------------------
# Phase 1 – Ersten Frame laden
# ---------------------------------------------------------------------------
from src.video_processing.video_reader import read_first_frame

logger.info("[Phase 1] Lade ersten Frame aus %s …", VIDEO)
frame0 = read_first_frame(VIDEO)
if frame0 is None:
    logger.error("Video konnte nicht geöffnet werden: %s", VIDEO)
    sys.exit(1)
logger.info("[Phase 1] Frame geladen: %dx%d px", frame0.shape[1], frame0.shape[0])

# ---------------------------------------------------------------------------
# Phase 2 – Video-Info + Homographie aus gespeicherter Kalibrierung
# ---------------------------------------------------------------------------
from src.video_processing.video_reader import get_video_info
from src.video_processing.calibration import load_calibration
from src.video_processing.homography import compute_homography, transform_points

logger.info("[Phase 2] Video-Info …")
info = get_video_info(VIDEO)
logger.info("[Phase 2] %dx%d px | %.1f fps | %d Frames gesamt",
            info.width, info.height, info.fps, info.frame_count)

H = None
try:
    src_pts, dst_pts = load_calibration(CALIB_JSON)
    H = compute_homography(src_pts, dst_pts)
    logger.info("[Phase 2] Homographie aus %s geladen ✓", CALIB_JSON)
except FileNotFoundError:
    logger.warning("[Phase 2] Keine Kalibrierungsdatei gefunden (%s). "
                   "Koordinaten-Transformation übersprungen.", CALIB_JSON)

# ---------------------------------------------------------------------------
# Phase 3 – YOLO-Detection
# ---------------------------------------------------------------------------
from src.object_detection.detector import Detector
from src.video_processing.video_reader import iter_frames

logger.info("[Phase 3] Lade YOLO-Modell (yolo11n.pt) …")
detector = Detector("yolo11n.pt", conf_thresholds={"player": 0.35, "ball": 0.35})

all_frame_detections: list = []

logger.info("[Phase 3] Verarbeite erste %d Frames …", N_FRAMES)
for fi, frame in iter_frames(VIDEO, max_frames=N_FRAMES):
    dets = detector.detect(frame, frame_index=fi)
    all_frame_detections.append(dets)

total_dets = sum(len(d) for d in all_frame_detections)
logger.info("[Phase 3] %d Detektionen in %d Frames", total_dets, N_FRAMES)

# ---------------------------------------------------------------------------
# Phase 4 – ByteTrack-Tracking
# ---------------------------------------------------------------------------
from src.tracking.tracker import PlayerTracker
from src.tracking.track import build_trajectories

logger.info("[Phase 4] ByteTrack-Tracking …")
tracker = PlayerTracker(frame_rate=int(info.fps))

all_tracked_frames = []
for fi, dets in enumerate(all_frame_detections):
    tracked = tracker.update(dets, frame_index=fi)
    all_tracked_frames.append(tracked)

trajectories = build_trajectories(all_tracked_frames)
logger.info("[Phase 4] %d eindeutige Track-IDs über %d Frames",
            len(trajectories), N_FRAMES)

if H is not None and trajectories:
    sample_id = next(iter(trajectories))
    traj = trajectories[sample_id]
    board_pts = transform_points(traj.points_px, H)
    logger.info("[Phase 4] Laufweg Spieler #%d: %d Punkte auf Taktikboard übertragen "
                "(Beispiel: erster Punkt = %.1f m, %.1f m)",
                sample_id, len(board_pts),
                board_pts[0][0] / 100, board_pts[0][1] / 100)

# ---------------------------------------------------------------------------
# Phase 5 – Teamzuordnung
# ---------------------------------------------------------------------------
from src.tracking.team_assigner import TeamAssigner, extract_hsv_feature, TEAM_A, TEAM_B

logger.info("[Phase 5] Teamzuordnung per HSV-K-Means …")
assigner = TeamAssigner()

# Features für alle Spieler des ersten Frames mit Detektionen sammeln
features_for_fit = []
for fi, (frame, tracked) in enumerate(
        zip(
            (f for _, f in iter_frames(VIDEO, max_frames=N_FRAMES)),
            all_tracked_frames,
        )
):
    for player in tracked:
        feat = extract_hsv_feature(frame, player.bbox)
        if feat is not None:
            features_for_fit.append(feat)
    if len(features_for_fit) >= 20:
        break

if len(features_for_fit) >= 2:
    assigner.fit(features_for_fit)
    logger.info("[Phase 5] K-Means trainiert auf %d Feature-Vektoren.", len(features_for_fit))

    # Zuordnung für das letzte Frame mit Tracking-Ergebnissen
    for fi, (frame, tracked) in enumerate(
            zip(
                (f for _, f in iter_frames(VIDEO, max_frames=N_FRAMES)),
                all_tracked_frames,
            )
    ):
        if not tracked:
            continue
        counts = {TEAM_A: 0, TEAM_B: 0}
        for player in tracked:
            feat = extract_hsv_feature(frame, player.bbox)
            if feat is not None:
                label = assigner.get_team(player.track_id, feat)
                counts[label] = counts.get(label, 0) + 1

    logger.info("[Phase 5] Letztes Frame: Team A=%d | Team B=%d Spieler",
                counts[TEAM_A], counts[TEAM_B])

    # Beispiel-Korrektur
    if trajectories:
        first_id = next(iter(trajectories))
        old = assigner.get_team(first_id,
                                features_for_fit[0])
        new_team = TEAM_B if old == TEAM_A else TEAM_A
        assigner.override(track_id=first_id, team=new_team)
        logger.info("[Phase 5] Manuelle Korrektur: Spieler #%d → Team %s",
                    first_id, "A" if new_team == TEAM_A else "B")
else:
    logger.warning("[Phase 5] Zu wenig Spieler erkannt für K-Means (%d Features).",
                   len(features_for_fit))

# ---------------------------------------------------------------------------
# Abschluss
# ---------------------------------------------------------------------------
logger.info("=== Demo abgeschlossen ===")
logger.info("Zusammenfassung:")
logger.info("  Video:       %s  (%dx%d, %.1f fps)",
            VIDEO, info.width, info.height, info.fps)
logger.info("  Frames:      %d analysiert", N_FRAMES)
logger.info("  Detektionen: %d", total_dets)
logger.info("  Trajectories:%d Spieler verfolgt", len(trajectories))
