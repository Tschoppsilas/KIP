"""Phase 4 tests: Spieler-Tracking mit ByteTrack.

Testet:
- TrackedPlayer Datenstruktur und center-Property
- Trajectory: add_point, smooth, build_trajectories
- PlayerTracker.update() mit gemocktem supervision.ByteTrack
- IDs bleiben über Frames konsistent
- Trajektorien sind auf das Taktikboard übertragbar (Should)
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(bbox=(10.0, 20.0, 50.0, 60.0), class_id=0,
                    class_name="player", confidence=0.9, frame_index=0):
    """Create a minimal DetectionResult-like object."""
    from src.object_detection.detection import DetectionResult
    return DetectionResult(
        bbox=bbox,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        frame_index=frame_index,
    )


# ---------------------------------------------------------------------------
# TrackedPlayer
# ---------------------------------------------------------------------------

class TestTrackedPlayer(unittest.TestCase):

    def test_center(self):
        from src.tracking.track import TrackedPlayer
        p = TrackedPlayer(track_id=1, bbox=(0.0, 0.0, 100.0, 80.0),
                          class_name="player", confidence=0.8)
        self.assertAlmostEqual(p.center[0], 50.0)
        self.assertAlmostEqual(p.center[1], 40.0)

    def test_width_height(self):
        from src.tracking.track import TrackedPlayer
        p = TrackedPlayer(track_id=2, bbox=(10.0, 20.0, 60.0, 90.0),
                          class_name="goalkeeper", confidence=0.7)
        self.assertAlmostEqual(p.width, 50.0)
        self.assertAlmostEqual(p.height, 70.0)

    def test_frame_index_default(self):
        from src.tracking.track import TrackedPlayer
        p = TrackedPlayer(track_id=3, bbox=(0, 0, 10, 10),
                          class_name="ball", confidence=0.5)
        self.assertEqual(p.frame_index, -1)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

class TestTrajectory(unittest.TestCase):

    def _make_traj(self, n=5):
        from src.tracking.track import Trajectory
        t = Trajectory(track_id=7, class_name="player")
        for i in range(n):
            t.add_point((float(i * 10), float(i * 5)), frame_index=i)
        return t

    def test_add_point_len(self):
        t = self._make_traj(4)
        self.assertEqual(len(t), 4)

    def test_add_point_with_board(self):
        from src.tracking.track import Trajectory
        t = Trajectory(track_id=1)
        t.add_point((100.0, 200.0), frame_index=0, center_board=(5.0, 10.0))
        self.assertEqual(t.points_board[0], (5.0, 10.0))

    def test_smooth_returns_same_length(self):
        t = self._make_traj(10)
        smoothed = t.smooth(window=3)
        self.assertEqual(len(smoothed), 10)

    def test_smooth_single_point(self):
        from src.tracking.track import Trajectory
        t = Trajectory(track_id=1)
        t.add_point((5.0, 5.0), frame_index=0)
        self.assertEqual(t.smooth(), [(5.0, 5.0)])

    def test_smooth_reduces_noise(self):
        """Smoothing of a noisy signal should approach the true mean."""
        from src.tracking.track import Trajectory
        rng = np.random.default_rng(42)
        t = Trajectory(track_id=1)
        for i in range(20):
            noise = float(rng.normal(0, 5))
            t.add_point((float(i) + noise, 0.0), frame_index=i)
        raw_std = np.std([p[0] for p in t.points_px])
        smooth_std = np.std([p[0] for p in t.smooth(window=5)])
        self.assertLess(smooth_std, raw_std)

    def test_frame_indices_tracked(self):
        t = self._make_traj(3)
        self.assertEqual(t.frame_indices, [0, 1, 2])


# ---------------------------------------------------------------------------
# build_trajectories
# ---------------------------------------------------------------------------

class TestBuildTrajectories(unittest.TestCase):

    def test_two_players_two_frames(self):
        from src.tracking.track import TrackedPlayer, build_trajectories
        frame0 = [
            TrackedPlayer(1, (0, 0, 10, 10), "player", 0.9, 0),
            TrackedPlayer(2, (20, 0, 30, 10), "player", 0.8, 0),
        ]
        frame1 = [
            TrackedPlayer(1, (5, 0, 15, 10), "player", 0.9, 1),
            TrackedPlayer(2, (25, 0, 35, 10), "player", 0.8, 1),
        ]
        trajs = build_trajectories([frame0, frame1])
        self.assertIn(1, trajs)
        self.assertIn(2, trajs)
        self.assertEqual(len(trajs[1]), 2)
        self.assertEqual(len(trajs[2]), 2)

    def test_empty_frames(self):
        from src.tracking.track import build_trajectories
        trajs = build_trajectories([[], []])
        self.assertEqual(trajs, {})

    def test_class_name_preserved(self):
        from src.tracking.track import TrackedPlayer, build_trajectories
        frame = [TrackedPlayer(5, (0, 0, 1, 1), "goalkeeper", 0.6, 0)]
        trajs = build_trajectories([frame])
        self.assertEqual(trajs[5].class_name, "goalkeeper")


# ---------------------------------------------------------------------------
# PlayerTracker (ByteTrack via supervision – gemockt)
# ---------------------------------------------------------------------------

class TestPlayerTracker(unittest.TestCase):
    """Tests for PlayerTracker using a mocked supervision.ByteTrack."""

    def _build_sv_detections(self, xyxy, confidence, tracker_ids):
        """Return a minimal supervision.Detections-like mock."""
        import supervision as sv
        det = MagicMock(spec=sv.Detections)
        det.xyxy = np.array(xyxy, dtype=np.float32)
        det.confidence = np.array(confidence, dtype=np.float32)
        det.tracker_id = np.array(tracker_ids, dtype=int)
        det.__len__ = lambda self: len(xyxy)
        return det

    def test_update_returns_tracked_players(self):
        from src.tracking.tracker import PlayerTracker
        xyxy = [[10.0, 20.0, 50.0, 60.0]]
        conf = [0.9]
        tids = [42]
        sv_result = self._build_sv_detections(xyxy, conf, tids)

        with patch("src.tracking.tracker.sv.ByteTrack") as MockBT:
            instance = MockBT.return_value
            instance.update_with_detections.return_value = sv_result
            tracker = PlayerTracker()
            detections = [_make_detection()]
            result = tracker.update(detections, frame_index=0)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].track_id, 42)
        self.assertAlmostEqual(result[0].confidence, 0.9)

    def test_consistent_ids_across_frames(self):
        """The same player should keep the same track_id across multiple frames."""
        from src.tracking.tracker import PlayerTracker

        def _sv_det_f0():
            return self._build_sv_detections(
                [[10.0, 20.0, 50.0, 60.0]], [0.9], [1])

        def _sv_det_f1():
            return self._build_sv_detections(
                [[12.0, 21.0, 52.0, 61.0]], [0.88], [1])

        results = []
        with patch("src.tracking.tracker.sv.ByteTrack") as MockBT:
            instance = MockBT.return_value
            instance.update_with_detections.side_effect = [_sv_det_f0(), _sv_det_f1()]
            tracker = PlayerTracker()
            for fi in range(2):
                r = tracker.update([_make_detection(frame_index=fi)], frame_index=fi)
                results.append(r)

        self.assertEqual(results[0][0].track_id, results[1][0].track_id)

    def test_empty_detection_list(self):
        from src.tracking.tracker import PlayerTracker
        import supervision as sv
        empty = sv.Detections.empty()

        with patch("src.tracking.tracker.sv.ByteTrack") as MockBT:
            instance = MockBT.return_value
            instance.update_with_detections.return_value = empty
            tracker = PlayerTracker()
            result = tracker.update([], frame_index=0)

        self.assertEqual(result, [])

    def test_reset_does_not_raise(self):
        from src.tracking.tracker import PlayerTracker
        with patch("src.tracking.tracker.sv.ByteTrack"):
            tracker = PlayerTracker()
            tracker.reset()  # should not raise

    # Should: Output passt zu Trajektorien-Input
    def test_tracked_output_feeds_trajectory(self):
        """TrackedPlayer objects returned by update() can build Trajectories."""
        from src.tracking.tracker import PlayerTracker
        from src.tracking.track import build_trajectories
        xyxy_seq = [
            [[10.0, 20.0, 50.0, 60.0]],
            [[12.0, 22.0, 52.0, 62.0]],
            [[14.0, 24.0, 54.0, 64.0]],
        ]
        with patch("src.tracking.tracker.sv.ByteTrack") as MockBT:
            instance = MockBT.return_value
            instance.update_with_detections.side_effect = [
                self._build_sv_detections(xy, [0.9], [7]) for xy in xyxy_seq
            ]
            tracker = PlayerTracker()
            all_frames = []
            for fi, _ in enumerate(xyxy_seq):
                players = tracker.update([_make_detection(frame_index=fi)], fi)
                all_frames.append(players)

        trajs = build_trajectories(all_frames)
        self.assertIn(7, trajs)
        self.assertEqual(len(trajs[7]), 3)
        # Should: Trajektorie auf Taktikboard übertragbar (Homographie anwendbar)
        from src.video_processing.homography import transform_points
        src_pts = [(0, 0), (100, 0), (100, 68), (0, 68)]
        dst_pts = [(0, 0), (10500, 0), (10500, 6800), (0, 6800)]
        from src.video_processing.homography import compute_homography
        H = compute_homography(src_pts, dst_pts)
        board_pts = transform_points(list(trajs[7].points_px), H)
        self.assertEqual(len(board_pts), 3)


if __name__ == "__main__":
    unittest.main()
