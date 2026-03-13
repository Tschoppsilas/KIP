"""Optional debugging helpers for visual frame inspection."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def show_frame_debug(frame_bgr: np.ndarray, title: str = "Debug Frame") -> Optional[object]:
    """Render a BGR frame with matplotlib for debugging purposes.

    Returns the matplotlib figure if matplotlib is installed, otherwise None.
    """
    if plt is None:
        return None

    frame_rgb = frame_bgr[:, :, ::-1]
    figure = plt.figure(figsize=(8, 4.5))
    axis = figure.add_subplot(111)
    axis.imshow(frame_rgb)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    return figure
