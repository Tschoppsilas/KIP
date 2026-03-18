"""Homography computation and point transformation for the tactic board."""

from __future__ import annotations

import numpy as np
import cv2


def compute_homography(
    src_points: list[tuple[float, float]],
    dst_points: list[tuple[float, float]],
) -> np.ndarray:
    """Compute a homography matrix from corresponding point pairs.

    Args:
        src_points: At least 4 points in video/image coordinates (x, y).
        dst_points: Corresponding points in tactic-board coordinates (x, y).

    Returns:
        3x3 homography matrix H such that dst ~ H * src (in homogeneous coords).

    Raises:
        ValueError: If fewer than 4 point pairs are provided or computation fails.
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        raise ValueError("At least 4 point pairs are required to compute a homography.")
    if len(src_points) != len(dst_points):
        raise ValueError("src_points and dst_points must have the same length.")

    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        raise RuntimeError("Homography computation failed (cv2.findHomography returned None).")
    return H


def transform_point(
    point: tuple[float, float],
    H: np.ndarray,
) -> tuple[float, float]:
    """Transform a single (x, y) point using homography matrix H.

    Returns:
        Transformed (x, y) in tactic-board coordinates.
    """
    src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)
    return float(dst[0][0][0]), float(dst[0][0][1])


def transform_points(
    points: list[tuple[float, float]],
    H: np.ndarray,
) -> list[tuple[float, float]]:
    """Transform a list of (x, y) points using homography matrix H."""
    if not points:
        return []
    src = np.array([[p] for p in points], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)
    return [(float(d[0][0]), float(d[0][1])) for d in dst]
