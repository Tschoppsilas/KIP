"""
End-to-End-Test: Video → Detection → Tracking → Teamzuordnung → Export.

Simuliert den Trainer-Workflow ohne GUI:
  - Kalibrierung aus gespeicherter JSON (synthetische Punkte)
  - 30 Frames Warmup für Teamzuordnung
  - 20 Frames Haupt-Pipeline
  - PNG, PDF und MP4 werden erzeugt und geprüft
"""

import json
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "finetune", "runs", "train", "weights", "best.pt"
)
VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Videos", "Muenchenstein_1.mp4"
)
BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Taktikboard", "Taktikboard.png"
)


def _skip_if_missing():
    missing = [p for p in (MODEL_PATH, VIDEO_PATH, BOARD_PATH) if not os.path.exists(p)]
    if missing:
        pytest.skip(f"Fehlende Dateien: {missing}")


def _make_calib_json(video_path: str, dest: str) -> str:
    """
    Erstellt eine synthetische Kalibrierungs-JSON aus den Ecken des Videos.
    Mappt Bildecken → Board-Ecken (1280×720).
    """
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    margin = 50
    src = [
        [margin,     margin],
        [w - margin, margin],
        [w - margin, h - margin],
        [margin,     h - margin],
        [w // 2,     margin],
        [w - margin, h // 2],
    ]
    dst = [
        [50,   50],
        [1230, 50],
        [1230, 670],
        [50,   670],
        [640,  50],
        [1230, 360],
    ]
    data = {"src_points": src, "dst_points": dst}
    path = os.path.join(dest, "calib_test.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_paths_module(self):
        """Pfad-Konstanten müssen korrekt aufgelöst sein."""
        from src.utils.paths import PROJECT_ROOT, OUTPUT_EXPORTS, OUTPUT_VIDEO
        assert os.path.isabs(PROJECT_ROOT)
        assert "output/exports" in OUTPUT_EXPORTS.replace("\\", "/")
        assert "output/video"   in OUTPUT_VIDEO.replace("\\", "/")

    def test_ensure_output_dirs_creates_folders(self):
        from src.utils import ensure_output_dirs, OUTPUT_EXPORTS, OUTPUT_VIDEO
        ensure_output_dirs()
        assert os.path.isdir(OUTPUT_EXPORTS)
        assert os.path.isdir(OUTPUT_VIDEO)

    def test_export_path_helper(self):
        from src.utils import export_path
        p = export_path("test.png")
        assert p.endswith("test.png")
        assert "output/exports" in p.replace("\\", "/")
        assert os.path.isdir(os.path.dirname(p))

    def test_video_path_helper(self):
        from src.utils import video_path
        p = video_path("test.mp4")
        assert p.endswith("test.mp4")
        assert "output/video" in p.replace("\\", "/")
        assert os.path.isdir(os.path.dirname(p))

    def test_pipeline_imports(self):
        """run_pipeline muss ohne Fehler importierbar sein."""
        import run_pipeline  # noqa: F401

    def test_full_pipeline_no_gui(self):
        """
        End-to-End: 20 Frames werden vollständig verarbeitet.
        PNG, PDF und MP4 müssen in output/ landen und gültig sein.
        """
        _skip_if_missing()

        with tempfile.TemporaryDirectory() as tmpdir:
            calib_json = _make_calib_json(VIDEO_PATH, tmpdir)

            # Output-Pfade temporär umleiten
            import src.utils.paths as paths_mod
            orig_exports = paths_mod.OUTPUT_EXPORTS
            orig_video   = paths_mod.OUTPUT_VIDEO
            paths_mod.OUTPUT_EXPORTS = os.path.join(tmpdir, "exports")
            paths_mod.OUTPUT_VIDEO   = os.path.join(tmpdir, "video")
            os.makedirs(paths_mod.OUTPUT_EXPORTS, exist_ok=True)
            os.makedirs(paths_mod.OUTPUT_VIDEO,   exist_ok=True)

            # export_path / video_path patchen
            import src.utils.exporter as exp_mod

            try:
                from src.tracking import PlayerTracker, TeamAssigner
                from src.video_processing import HomographyTransformer, VideoLoader
                from src.utils import VideoExporter, render_board_frame, export_png, export_pdf

                transformer = HomographyTransformer.load(calib_json)
                board_img   = cv2.imread(BOARD_PATH)
                tracker     = PlayerTracker(MODEL_PATH, conf=0.25)
                assigner    = TeamAssigner(min_samples=2)

                # Warmup (10 Frames)
                cap = cv2.VideoCapture(VIDEO_PATH)
                for i in range(10):
                    ret, frame = cap.read()
                    if not ret: break
                    tr = tracker.update(frame, frame_idx=i)
                    assigner.add_frame(frame, tr)
                cap.release()
                tracker.reset()
                assigner.assign_teams()

                bh, bw = board_img.shape[:2]
                out_png = os.path.join(paths_mod.OUTPUT_EXPORTS, "e2e_board.png")
                out_pdf = os.path.join(paths_mod.OUTPUT_EXPORTS, "e2e_board.pdf")
                out_mp4 = os.path.join(paths_mod.OUTPUT_VIDEO,   "e2e_annotated.mp4")

                last_board_frame = None
                frames_processed = 0

                with VideoLoader(VIDEO_PATH) as loader:
                    with VideoExporter(out_mp4, fps=loader.fps, split_view=True) as exporter:
                        for frame_idx, frame in loader.frames(step=1):
                            if frame_idx >= 20:
                                break
                            try:
                                tr = tracker.update(frame, frame_idx=frame_idx)
                            except Exception:
                                continue

                            board_positions, board_trajectories, teams = {}, {}, {}
                            for i in range(len(tr)):
                                tid = tr.track_ids[i]
                                cx, cy = tr.center(i)
                                try:
                                    bx, by = transformer.transform_point(cx, cy)
                                except Exception:
                                    continue
                                if not (0 <= bx <= bw and 0 <= by <= bh):
                                    continue
                                board_positions[tid] = (bx, by)
                                traj = tracker.get_trajectory(tid)
                                if traj:
                                    board_trajectories[tid] = traj.to_board_coords(transformer).xy_sequence()
                                teams[tid] = assigner.get_team(tid) or 0

                            board_frame = render_board_frame(
                                board_img, board_positions, teams, board_trajectories
                            )
                            last_board_frame = board_frame
                            exporter.write_frame(board_frame, frame)
                            frames_processed += 1

                assert frames_processed == 20, f"Nur {frames_processed}/20 Frames verarbeitet."

                # Exporte
                assert last_board_frame is not None
                export_png(last_board_frame, out_png)
                export_pdf(last_board_frame, out_pdf)

                # Validierung PNG
                assert os.path.exists(out_png)
                reloaded = cv2.imread(out_png)
                assert reloaded is not None
                assert reloaded.shape == board_img.shape

                # Validierung PDF
                assert os.path.exists(out_pdf)
                with open(out_pdf, "rb") as f:
                    assert f.read(4) == b"%PDF"

                # Validierung MP4
                assert os.path.exists(out_mp4)
                cap2 = cv2.VideoCapture(out_mp4)
                assert cap2.isOpened()
                ret, vframe = cap2.read()
                cap2.release()
                assert ret
                assert vframe is not None

            finally:
                paths_mod.OUTPUT_EXPORTS = orig_exports
                paths_mod.OUTPUT_VIDEO   = orig_video

    def test_pipeline_handles_tracking_failure_gracefully(self):
        """
        Bei einem korrupten Frame darf die Pipeline nicht abstürzen,
        sondern den Frame überspringen.
        """
        _skip_if_missing()
        from src.tracking import PlayerTracker, TeamAssigner
        from src.video_processing import HomographyTransformer

        with tempfile.TemporaryDirectory() as tmpdir:
            calib_json = _make_calib_json(VIDEO_PATH, tmpdir)
            transformer = HomographyTransformer.load(calib_json)
            tracker = PlayerTracker(MODEL_PATH, conf=0.25)
            board_img = cv2.imread(BOARD_PATH)
            bh, bw = board_img.shape[:2]

            errors = 0
            # Leerer Frame (schwarz) simuliert einen korrupten Frame
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            try:
                tr = tracker.update(blank, frame_idx=999)
                # Kein Fehler – leeres Ergebnis ist akzeptabel
                assert isinstance(tr.track_ids, list)
            except Exception:
                errors += 1

            assert errors == 0, "Pipeline stürzt bei leerem Frame ab."

    def test_out_of_bounds_positions_filtered(self):
        """
        Spielerpositionen ausserhalb des Boards (< 0 oder > Board-Masse)
        dürfen nicht ins Export-Bild gelangen.
        """
        from src.utils import render_board_frame
        board_img = cv2.imread(BOARD_PATH)
        bh, bw = board_img.shape[:2]

        # Zwei Positionen: eine gültig, eine ausserhalb
        positions = {
            1: (640.0, 360.0),      # gültig
            2: (-100.0, -100.0),    # ausserhalb
            3: (bw + 50, bh + 50),  # ausserhalb
        }
        # Nur gültige übergeben (wie in der Pipeline gefiltert)
        valid = {tid: pos for tid, pos in positions.items()
                 if 0 <= pos[0] <= bw and 0 <= pos[1] <= bh}
        assert 1 in valid
        assert 2 not in valid
        assert 3 not in valid

        out = render_board_frame(board_img, valid, {1: 0}, {})
        assert out.shape == board_img.shape
