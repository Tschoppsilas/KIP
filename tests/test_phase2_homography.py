"""Tests for Phase 2 homography computation and point transformation."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_processing.homography import compute_homography, transform_point, transform_points


_SRC = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
_DST = [(0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)]


class TestComputeHomography(unittest.TestCase):
    def test_returns_3x3_matrix(self) -> None:
        H = compute_homography(_SRC, _DST)
        self.assertEqual(H.shape, (3, 3))

    def test_raises_for_fewer_than_4_points(self) -> None:
        with self.assertRaises(ValueError):
            compute_homography(_SRC[:3], _DST[:3])

    def test_raises_for_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            compute_homography(_SRC, _DST[:3])


class TestTransformPoint(unittest.TestCase):
    def setUp(self) -> None:
        self.H = compute_homography(_SRC, _DST)

    def test_corner_maps_correctly(self) -> None:
        result = transform_point((0.0, 0.0), self.H)
        self.assertAlmostEqual(result[0], 0.0, places=3)
        self.assertAlmostEqual(result[1], 0.0, places=3)

    def test_opposite_corner_maps_correctly(self) -> None:
        result = transform_point((100.0, 100.0), self.H)
        self.assertAlmostEqual(result[0], 40.0, places=3)
        self.assertAlmostEqual(result[1], 20.0, places=3)

    def test_center_maps_to_center(self) -> None:
        result = transform_point((50.0, 50.0), self.H)
        self.assertAlmostEqual(result[0], 20.0, places=3)
        self.assertAlmostEqual(result[1], 10.0, places=3)


class TestTransformPoints(unittest.TestCase):
    def setUp(self) -> None:
        self.H = compute_homography(_SRC, _DST)

    def test_empty_list_returns_empty(self) -> None:
        self.assertEqual(transform_points([], self.H), [])

    def test_all_corners_map_correctly(self) -> None:
        results = transform_points(_SRC, self.H)
        for result, expected in zip(results, _DST):
            self.assertAlmostEqual(result[0], expected[0], places=3)
            self.assertAlmostEqual(result[1], expected[1], places=3)
