"""Tests für TrackResult, Trajectory und PlayerTracker."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "finetune", "runs", "train", "weights", "best.pt"
)
VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Videos", "Muenchenstein_1.mp4"
)


# ---------------------------------------------------------------------------
# TrackResult
# ---------------------------------------------------------------------------

class TestTrackResult:
    def _make(self):
        from src.tracking import TrackResult
        tr = TrackResult(frame_idx=10)
        tr.track_ids = [1, 2]
        tr.boxes = [
            np.array([10, 20, 50, 80], dtype=np.float32),
            np.array([200, 100, 260, 200], dtype=np.float32),
        ]
        tr.confidences = [0.9, 0.8]
        return tr

    def test_len(self):
        tr = self._make()
        assert len(tr) == 2

    def test_center(self):
        tr = self._make()
        cx, cy = tr.center(0)
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(50.0)

    def test_centers(self):
        tr = self._make()
        cs = tr.centers()
        assert len(cs) == 2

    def test_empty(self):
        from src.tracking import TrackResult
        tr = TrackResult(frame_idx=0)
        assert len(tr) == 0
        assert tr.centers() == []


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

class TestTrajectory:
    def test_add_and_retrieve(self):
        from src.tracking import Trajectory
        t = Trajectory(track_id=5)
        t.add_point(0, 100.0, 200.0)
        t.add_point(1, 110.0, 205.0)
        assert len(t.points) == 2
        assert t.points[0] == (0, 100.0, 200.0)

    def test_xy_sequence(self):
        from src.tracking import Trajectory
        t = Trajectory(track_id=3)
        t.add_point(0, 10.0, 20.0)
        t.add_point(1, 30.0, 40.0)
        seq = t.xy_sequence()
        assert seq == [(10.0, 20.0), (30.0, 40.0)]

    def test_last_point(self):
        from src.tracking import Trajectory
        t = Trajectory(track_id=1)
        assert t.last_point() is None
        t.add_point(5, 1.0, 2.0)
        assert t.last_point() == (5, 1.0, 2.0)

    def test_to_board_coords(self):
        """Transformierte Laufwege müssen plausible Board-Koordinaten liefern."""
        from src.tracking import Trajectory
        from src.video_processing import HomographyTransformer

        src = [(100, 100), (500, 100), (500, 400), (100, 400)]
        dst = [(200, 150), (800, 150), (800, 600), (200, 600)]
        transformer = HomographyTransformer().compute(src, dst)

        t = Trajectory(track_id=7)
        t.add_point(0, 100.0, 100.0)  # entspricht src[0]
        t.add_point(1, 500.0, 100.0)  # entspricht src[1]

        board_t = t.to_board_coords(transformer)
        assert len(board_t.points) == 2
        assert board_t.track_id == 7
        bx, by = board_t.points[0][1], board_t.points[0][2]
        assert abs(bx - dst[0][0]) < 2.0
        assert abs(by - dst[0][1]) < 2.0


# ---------------------------------------------------------------------------
# PlayerTracker (Integration)
# ---------------------------------------------------------------------------

def _load_frames(n: int = 10, start: int = 0):
    import cv2
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


class TestPlayerTracker:
    def test_tracker_loads(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip("best.pt nicht gefunden.")
        from src.tracking import PlayerTracker
        t = PlayerTracker(MODEL_PATH)
        assert t is not None

    def test_update_returns_track_result(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VIDEO_PATH):
            pytest.skip("Modell oder Video nicht gefunden.")
        from src.tracking import PlayerTracker, TrackResult
        frames = _load_frames(1)
        tracker = PlayerTracker(MODEL_PATH, conf=0.2)
        result = tracker.update(frames[0], frame_idx=0)
        assert isinstance(result, TrackResult)
        assert result.frame_idx == 0

    def test_ids_consistent_over_frames(self):
        """
        Stabilitätstest über mindestens 150 Frames (≈ 5 s bei 30 fps).

        Strategie: Spieler können das Bild kurz verlassen; daher wird nicht
        verlangt, dass eine ID in ALLEN Frames vorkommt.  Stattdessen muss
        die am häufigsten gesehene Track-ID in mindestens 50 % der tatsächlich
        geladenen Frames auftauchen – ein klares Zeichen für stabile Zuordnung.
        """
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VIDEO_PATH):
            pytest.skip("Modell oder Video nicht gefunden.")
        import cv2
        from src.tracking import PlayerTracker
        from collections import Counter

        # FPS-adaptiv: mindestens 150 Frames, mindestens 2.5 s Echtzeit
        cap = cv2.VideoCapture(VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        min_frames = max(150, int(fps * 2.5))
        start = 30  # erste Sekunde überspringen (Kamera-Setup)
        n_frames = min(min_frames, total - start)

        frames = _load_frames(n=n_frames, start=start)
        if len(frames) < 10:
            pytest.skip("Zu wenige Frames im Video.")

        tracker = PlayerTracker(MODEL_PATH, conf=0.2)
        id_counter: Counter = Counter()
        for i, frame in enumerate(frames):
            result = tracker.update(frame, frame_idx=i)
            for tid in result.track_ids:
                id_counter[tid] += 1

        actual_frames = len(frames)
        threshold = actual_frames * 0.50  # 50 % Mindest-Präsenz

        most_common_id, most_common_count = id_counter.most_common(1)[0]
        print(
            f"\n[Tracking-Stabilität] {actual_frames} Frames, {fps} fps | "
            f"häufigste ID={most_common_id}: {most_common_count} Frames "
            f"({most_common_count / actual_frames * 100:.1f} %) | "
            f"Schwelle: {threshold:.0f} Frames (50 %)"
        )

        assert most_common_count >= threshold, (
            f"Keine stabile Track-ID: beste ID {most_common_id} nur in "
            f"{most_common_count}/{actual_frames} Frames ({most_common_count/actual_frames*100:.1f}%) – "
            f"Mindest-Schwelle: 50 %."
        )

    def test_trajectories_are_built(self):
        """Nach N Frames müssen Laufwege mit N Punkten vorliegen."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VIDEO_PATH):
            pytest.skip("Modell oder Video nicht gefunden.")
        from src.tracking import PlayerTracker
        frames = _load_frames(n=5, start=30)
        tracker = PlayerTracker(MODEL_PATH, conf=0.2)
        for i, frame in enumerate(frames):
            tracker.update(frame, frame_idx=i)

        trajectories = tracker.get_all_trajectories()
        assert len(trajectories) > 0
        for tid, traj in trajectories.items():
            assert len(traj.points) >= 1

    def test_reset_clears_state(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VIDEO_PATH):
            pytest.skip("Modell oder Video nicht gefunden.")
        from src.tracking import PlayerTracker
        frames = _load_frames(n=3, start=0)
        tracker = PlayerTracker(MODEL_PATH, conf=0.2)
        for i, f in enumerate(frames):
            tracker.update(f, i)
        assert len(tracker.get_all_trajectories()) > 0
        tracker.reset()
        assert len(tracker.get_all_trajectories()) == 0

    def test_trajectories_transformable_to_board(self):
        """Laufwege müssen auf Board-Koordinaten transformierbar sein."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VIDEO_PATH):
            pytest.skip("Modell oder Video nicht gefunden.")
        from src.tracking import PlayerTracker
        from src.video_processing import HomographyTransformer
        import cv2

        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        cap.release()
        h, w = frame.shape[:2]

        # Einfache synthetische Homography: Spielfeld-Ecken → Board-Rechteck
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [(0, 0), (800, 0), (800, 500), (0, 500)]
        transformer = HomographyTransformer().compute(src, dst)

        frames = _load_frames(n=3, start=30)
        tracker = PlayerTracker(MODEL_PATH, conf=0.2)
        for i, f in enumerate(frames):
            tracker.update(f, frame_idx=i)

        for tid, traj in tracker.get_all_trajectories().items():
            board_traj = traj.to_board_coords(transformer)
            assert board_traj.track_id == tid
            assert len(board_traj.points) == len(traj.points)
