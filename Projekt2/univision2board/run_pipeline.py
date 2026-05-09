"""
UniVision2Board – Haupt-Pipeline (Post-Game-Analyse)

Verwendung:
    source .venv/bin/activate
    python run_pipeline.py Videos/Muenchenstein_1.mp4

Optionen:
    --model       Pfad zum YOLO-Gewicht (Standard: finetune/runs/train/weights/best.pt)
    --calib       Pfad zur Kalibrierungs-JSON (Standard: output/exports/calibration_<Video>.json)
    --auto-calib  Ohne GUI: Videoecken → Board automatisch abgleichen (falls keine JSON existiert)
    --conf        YOLO-Confidence-Schwelle (Standard: 0.25)
    --step        Jeden N-ten Frame verarbeiten (Standard: 1 = alle)
    --no-gui      Kein BoardWindow; keine Kalibrierungs-GUI — Kalibrierungsdatei muss existieren
                  oder --auto-calib setzen

Ausgabe:
    output/exports/<videoname>_board.png   – Standbild letzter Frame
    output/exports/<videoname>_board.pdf   – PDF-Version
    output/video/<videoname>_annotated.mp4 – Annotiertes Split-View-Video
"""

import argparse
import json
import os
import sys
import time

import cv2

# Projektpfad sicherstellen
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _fix_qt_plugin_path() -> None:
    """
    OpenCV setzt oft QT_QPA_PLATFORM_PLUGIN_PATH auf cv2/qt/plugins —
    das bricht PyQt5 (xcb konnte nicht geladen werden).
    PyQt5-eigene Plugins erzwingen.
    """
    try:
        import pathlib
        import PyQt5
        root = pathlib.Path(PyQt5.__file__).resolve().parent
        for cand in (root / "Qt5" / "plugins", root / "plugins"):
            if (cand / "platforms").is_dir():
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(cand)
                break
    except Exception:
        pass

from src.tracking import PlayerTracker, TeamAssigner
from src.video_processing import HomographyTransformer, VideoLoader
from src.utils import (
    VideoExporter, export_pdf, export_png, get_logger,
    render_board_frame, ensure_output_dirs, export_path, video_path,
)

logger = get_logger("pipeline")

# ---------------------------------------------------------------------------
# Pfad-Defaults (relativ zum Projektroot)
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL  = os.path.join(_ROOT, "..", "finetune", "runs", "train", "weights", "best.pt")
_DEFAULT_BOARD  = os.path.join(_ROOT, "..", "Taktikboard", "Taktikboard.png")
_CALIB_DIR      = os.path.join(_ROOT, "..", "output", "exports")


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _calib_path(video_path_arg: str) -> str:
    """Ableiteter JSON-Pfad für Kalibrierung: output/exports/<videoname>.json"""
    name = os.path.splitext(os.path.basename(video_path_arg))[0]
    os.makedirs(_CALIB_DIR, exist_ok=True)
    return os.path.join(_CALIB_DIR, f"calibration_{name}.json")


def _write_auto_calibration(video_path_arg: str, calib_json: str) -> None:
    """
    Schreibt eine grobe Homography ohne GUI: Videocken → Board-Ecken (1280×720).

    Qualität ist eingeschränkt — für echte Analyse Kalibrierungs-GUI verwenden.
    """
    cap = cv2.VideoCapture(video_path_arg)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    margin = 50
    src = [
        [margin, margin],
        [w - margin, margin],
        [w - margin, h - margin],
        [margin, h - margin],
        [w // 2, margin],
        [w - margin, h // 2],
    ]
    dst = [
        [50, 50],
        [1230, 50],
        [1230, 670],
        [50, 670],
        [640, 50],
        [1230, 360],
    ]
    data = {"src_points": src, "dst_points": dst}
    os.makedirs(os.path.dirname(os.path.abspath(calib_json)), exist_ok=True)
    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Automatische Kalibrierung geschrieben: {calib_json}")


def _load_or_calibrate(
    video_path_arg: str,
    calib_json: str,
    *,
    interactive_calibration: bool,
    auto_calib: bool,
) -> HomographyTransformer:
    """
    Gibt immer eine gültige HomographyTransformer zurück.

    GUI-Modus (interactive_calibration=True):
        Der Kalibrierungs-Dialog wird IMMER geöffnet.
        Falls eine frühere JSON existiert, kann der Trainer sie im Dialog
        über "Kalibrierung laden …" wiederverwenden — er wird aber nie
        stillschweigend übergangen.

    Headless-Modus (interactive_calibration=False):
        --calib <pfad>  → explizite JSON wird direkt geladen.
        --auto-calib    → grobe Ecken-Homography ohne GUI (nur für Tests).
        Sonst           → Fehler mit Hinweis.
    """
    if not interactive_calibration:
        if os.path.exists(calib_json):
            logger.info(f"Kalibrierung geladen (headless): {calib_json}")
            return HomographyTransformer.load(calib_json)
        if auto_calib:
            _write_auto_calibration(video_path_arg, calib_json)
            return HomographyTransformer.load(calib_json)
        raise RuntimeError(
            "Headless-Betrieb erfordert eine Kalibrierungsdatei.\n"
            f"  Erwartet: {calib_json}\n"
            "  Optionen:\n"
            "    1) Einmal mit GUI kalibrieren (ohne --no-gui) → JSON wird gespeichert\n"
            "    2) --calib <pfad>  um eine bestehende JSON explizit anzugeben\n"
            "    3) --auto-calib   für eine grobe Videoecken-Zuordnung (nur Tests)"
        )

    # GUI-Modus: Dialog wird IMMER gezeigt.
    # Die zuvor gespeicherte JSON kann über "Kalibrierung laden …" im Dialog
    # wiederverwendet werden, wird aber nie automatisch übersprungen.
    logger.info("Kalibrierungs-Dialog wird geöffnet (zwingend bei jedem Start).")
    _fix_qt_plugin_path()

    board_img = cv2.imread(_DEFAULT_BOARD)
    if board_img is None:
        raise FileNotFoundError(f"Taktikboard nicht gefunden: {_DEFAULT_BOARD}")

    with VideoLoader(video_path_arg) as loader:
        frame = loader.read_frame(0)
    if frame is None:
        raise RuntimeError("Erster Frame konnte nicht gelesen werden.")

    import sys as _sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(_sys.argv)

    from src.gui import CalibrationDialog

    dlg = CalibrationDialog(frame, board_img, calib_json)
    if dlg.exec_() != CalibrationDialog.Accepted:
        raise RuntimeError("Kalibrierung abgebrochen – Pipeline wird nicht gestartet.")

    transformer = dlg.get_transformer()
    if transformer is None:
        raise RuntimeError("Keine Homography berechnet – Pipeline wird nicht gestartet.")

    logger.info(f"Kalibrierung gespeichert: {calib_json}")
    return transformer


def _run_team_calibration(
    tracker: PlayerTracker,
    assigner: TeamAssigner,
    video_path_arg: str,
    n_warmup: int = 30,
) -> None:
    """
    Akkumuliert Farbproben aus den ersten n_warmup Frames und berechnet K-Means.
    Bei Bedarf öffnet sich der TeamCorrectionDialog.
    """
    logger.info(f"Team-Warmup: {n_warmup} Frames ...")
    cap = cv2.VideoCapture(video_path_arg)
    for i in range(n_warmup):
        ret, frame = cap.read()
        if not ret:
            break
        tr = tracker.update(frame, frame_idx=i)
        assigner.add_frame(frame, tr)
    cap.release()
    tracker.reset()  # Tracking-State zurück – Produktion startet sauber

    assignments = assigner.assign_teams()
    if len(assignments) == 0:
        logger.warning("Keine Spieler für Teamzuordnung gefunden – alle werden Team A zugewiesen.")
        return

    teams_found = set(assignments.values())
    if len(teams_found) < 2:
        logger.warning(
            "K-Means hat nur ein Team erkannt – "
            "manuelle Korrekturen über GUI empfohlen."
        )
    else:
        logger.info(
            f"Team-Zuweisung: {sum(v==0 for v in assignments.values())} × Team A, "
            f"{sum(v==1 for v in assignments.values())} × Team B"
        )


# ---------------------------------------------------------------------------
# Haupt-Pipeline
# ---------------------------------------------------------------------------

def run(
    video_path_arg: str,
    model_path: str,
    calib_json: str,
    conf: float = 0.25,
    step: int = 1,
    show_gui: bool = True,
    interactive_calibration: bool = True,
    auto_calib: bool = False,
) -> None:
    ensure_output_dirs()
    video_name = os.path.splitext(os.path.basename(video_path_arg))[0]

    # 1) Kalibrierung
    transformer = _load_or_calibrate(
        video_path_arg,
        calib_json,
        interactive_calibration=interactive_calibration,
        auto_calib=auto_calib,
    )

    # 2) Tracker + Assigner initialisieren
    tracker = PlayerTracker(model_path, conf=conf)
    assigner = TeamAssigner(min_samples=3)

    # 3) Team-Warmup (Farbproben sammeln, K-Means)
    _run_team_calibration(tracker, assigner, video_path_arg, n_warmup=30)

    # 4) Board-Hintergrundbild
    board_img = cv2.imread(_DEFAULT_BOARD)
    if board_img is None:
        raise FileNotFoundError(f"Taktikboard nicht gefunden: {_DEFAULT_BOARD}")

    # 5) GUI-Fenster (optional)
    board_window = None
    app = None
    if show_gui:
        _fix_qt_plugin_path()
        import sys as _sys
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(_sys.argv)
        from src.gui import BoardWindow

        board_window = BoardWindow(_DEFAULT_BOARD)
        board_window.set_assigner(assigner)
        board_window.show()

    # 6) Export-Pfade
    out_png  = export_path(f"{video_name}_board.png")
    out_pdf  = export_path(f"{video_name}_board.pdf")
    out_mp4  = video_path(f"{video_name}_annotated.mp4")

    # 7) Haupt-Verarbeitungsschleife
    last_board_frame = None

    with VideoLoader(video_path_arg) as loader:
        fps = loader.fps or 30.0
        total = loader.frame_count
        logger.info(f"Starte Verarbeitung: {total} Frames @ {fps:.1f} fps, step={step}")

        with VideoExporter(out_mp4, fps=fps / step, split_view=True) as exporter:
            for frame_idx, frame in loader.frames(step=step):
                # Pause-Spin-Wait (blockiert Schleife bis Video fortgesetzt wird)
                if board_window is not None and app is not None:
                    while board_window.is_paused():
                        time.sleep(0.05)
                        app.processEvents()

                # Detection + Tracking
                try:
                    track_result = tracker.update(frame, frame_idx=frame_idx)
                except Exception as e:
                    logger.warning(f"Frame {frame_idx}: Tracking-Fehler – {e}")
                    continue

                # Gelöschte Spieler aus dem Frame herausfiltern
                if board_window is not None:
                    track_result = track_result.filtered(board_window.deleted_ids)

                # Positionen auf Board transformieren
                board_positions = {}
                board_trajectories = {}
                for i in range(len(track_result)):
                    tid = track_result.track_ids[i]
                    cx, cy = track_result.center(i)
                    try:
                        bx, by = transformer.transform_point(cx, cy)
                    except Exception:
                        continue
                    board_positions[tid] = (bx, by)

                    traj = tracker.get_trajectory(tid)
                    if traj is not None:
                        board_trajectories[tid] = traj.to_board_coords(transformer).xy_sequence()

                # Teamzuordnung: alle Track-IDs im Frame (für Video-Overlay-Farben)
                teams_overlay = {}
                for i in range(len(track_result)):
                    tid = track_result.track_ids[i]
                    tm = assigner.get_team(tid)
                    teams_overlay[tid] = tm if tm is not None else 0

                # Ausreisser filtern: Punkte ausserhalb des Boards verwerfen
                bh, bw = board_img.shape[:2]
                board_positions = {
                    tid: pos for tid, pos in board_positions.items()
                    if 0 <= pos[0] <= bw and 0 <= pos[1] <= bh
                }

                teams = {tid: teams_overlay[tid] for tid in board_positions}

                # Board rendern
                board_frame = render_board_frame(
                    board_img, board_positions, teams, board_trajectories
                )
                last_board_frame = board_frame

                # GUI: Live-Split (Video + Boxes | Taktikboard)
                if board_window is not None and app is not None:
                    board_window.update_live_split(
                        frame,
                        track_result,
                        teams_overlay,
                        board_positions,
                        board_trajectories,
                        [],
                    )
                    app.processEvents()

                # Video-Frame schreiben
                exporter.write_frame(board_frame, frame)

                if frame_idx % 100 == 0:
                    logger.info(f"  … Frame {frame_idx}/{total}")

    # 8) Standbild exportieren
    if last_board_frame is not None:
        export_png(last_board_frame, out_png)
        export_pdf(last_board_frame, out_pdf)
        logger.info(f"Exportiert: {out_png}")
        logger.info(f"Exportiert: {out_pdf}")

    logger.info(f"Video exportiert: {out_mp4}")
    logger.info("Pipeline abgeschlossen.")

    if board_window is not None and app is not None:
        app.exec_()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniVision2Board Pipeline")
    parser.add_argument("video", help="Pfad zur Videodatei")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="YOLO-Gewicht (.pt)")
    parser.add_argument("--calib", default=None, help="Kalibrierungs-JSON")
    parser.add_argument(
        "--auto-calib",
        action="store_true",
        help="Ohne GUI: grobe Eck-Zuordnung schreiben wenn keine Kalibrierungsdatei existiert",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence-Schwelle")
    parser.add_argument("--step", type=int, default=1, help="Jeden N-ten Frame")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Kein Taktikboard-Fenster; keine Kalibrierungs-GUI (Kalibrierungsdatei oder --auto-calib)",
    )
    args = parser.parse_args()

    calib = args.calib or _calib_path(args.video)

    run(
        video_path_arg=args.video,
        model_path=args.model,
        calib_json=calib,
        conf=args.conf,
        step=args.step,
        show_gui=not args.no_gui,
        interactive_calibration=not args.no_gui,
        auto_calib=args.auto_calib,
    )
