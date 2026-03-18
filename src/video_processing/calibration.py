"""Saving, loading, and managing calibration points for homography."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .homography import compute_homography

CalibrationPoints = list[tuple[float, float]]

BOARD_WIDTH = 40.0
BOARD_HEIGHT = 20.0

DEFAULT_BOARD_CORNERS: CalibrationPoints = [
    (0.0, 0.0),
    (BOARD_WIDTH, 0.0),
    (BOARD_WIDTH, BOARD_HEIGHT),
    (0.0, BOARD_HEIGHT),
]


def save_calibration(
    src_points: CalibrationPoints,
    dst_points: CalibrationPoints,
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Persist calibration point pairs to a JSON file.

    Args:
        src_points: Video/image coordinates (x, y).
        dst_points: Corresponding tactic-board coordinates (x, y).
        path: Destination file path (.json).
        metadata: Optional dict with extra info (e.g. video name, date).
    """
    data: dict = {
        "src_points": [list(p) for p in src_points],
        "dst_points": [list(p) for p in dst_points],
    }
    if metadata:
        data["metadata"] = metadata

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> tuple[CalibrationPoints, CalibrationPoints]:
    """Load calibration point pairs from a JSON file.

    Returns:
        Tuple (src_points, dst_points).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    src = [tuple(p) for p in data["src_points"]]
    dst = [tuple(p) for p in data["dst_points"]]
    return src, dst  # type: ignore[return-value]


def load_homography_from_file(path: str | Path) -> np.ndarray:
    """Load calibration from file and return the computed homography matrix."""
    src, dst = load_calibration(path)
    return compute_homography(src, dst)


def save_template(
    template_name: str,
    src_points: CalibrationPoints,
    dst_points: CalibrationPoints,
    templates_dir: str | Path = "calibration_templates",
) -> Path:
    """Save a reusable calibration template for a recurring camera angle.

    Templates are stored as named JSON files inside *templates_dir*.

    Returns:
        Path to the written template file.
    """
    out_path = Path(templates_dir) / f"{template_name}.json"
    save_calibration(src_points, dst_points, out_path, metadata={"template": template_name})
    return out_path


def load_template(
    template_name: str,
    templates_dir: str | Path = "calibration_templates",
) -> tuple[CalibrationPoints, CalibrationPoints]:
    """Load a previously saved calibration template by name."""
    path = Path(templates_dir) / f"{template_name}.json"
    return load_calibration(path)


def list_templates(templates_dir: str | Path = "calibration_templates") -> list[str]:
    """Return a list of available template names in *templates_dir*."""
    d = Path(templates_dir)
    if not d.exists():
        return []
    return [p.stem for p in sorted(d.glob("*.json"))]
