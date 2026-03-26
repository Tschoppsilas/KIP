"""OpenCV-based GUI dialog for selecting calibration points on a video frame.

Usage (interactive, requires a display):
    from src.gui.calibration_dialog import collect_calibration_points
    src_pts = collect_calibration_points(frame_bgr, n_points=8)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_POINT_COLOR = (0, 255, 0)
_POINT_RADIUS = 6
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_FONT_THICKNESS = 2


def _draw_points(canvas: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    overlay = canvas.copy()
    for idx, (px, py) in enumerate(points):
        cv2.circle(overlay, (px, py), _POINT_RADIUS, _POINT_COLOR, -1)
        cv2.putText(
            overlay,
            str(idx + 1),
            (px + 8, py - 8),
            _FONT,
            _FONT_SCALE,
            _POINT_COLOR,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return overlay


def collect_calibration_points(
    frame_bgr: np.ndarray,
    n_points: int = 8,
    window_title: str = "Calibration - Click field points (Enter=confirm, ESC=abort)",
    max_display_width: int = 1280,
) -> list[tuple[float, float]] | None:
    """Open an OpenCV window and let the user click n_points on the frame.

    Args:
        frame_bgr: The video frame to display as background.
        n_points: Number of points to collect (6–8 recommended).
        window_title: Title shown on the OpenCV window.
        max_display_width: Downscale the display if the frame is wider than this.

    Returns:
        List of (x, y) pixel coordinates in original frame resolution,
        or None if the user pressed ESC.
    """
    scale = 1.0
    display = frame_bgr.copy()
    h, w = display.shape[:2]
    if w > max_display_width:
        scale = max_display_width / w
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

    clicked: list[tuple[int, int]] = []

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < n_points:
            clicked.append((x, y))
            logger.debug("Punkt %d geklickt: (%d, %d)", len(clicked), x, y)

    try:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        callback_set = False
        for attempt in range(1, 4):
            cv2.imshow(window_title, _draw_points(display.copy(), clicked))
            cv2.waitKey(1)  # Event-Pump fuer stabile Fensterinitialisierung
            visible = float(cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE))
            if visible < 1.0:
                continue
            try:
                cv2.setMouseCallback(window_title, _on_mouse)
                callback_set = True
                break
            except cv2.error:
                pass
        if not callback_set:
            raise RuntimeError(
                "Kalibrierungsfenster konnte nicht stabil initialisiert werden "
                "(OpenCV Mouse-Callback)."
            )
    except cv2.error as exc:
        cv2.destroyAllWindows()
        raise RuntimeError(
            "Kalibrierungsfenster konnte nicht geoeffnet werden. "
            "Kein Display verfuegbar oder OpenCV-Backend nicht unterstuetzt. "
            f"Details: {exc}"
        ) from exc

    logger.info(
        "Kalibrierungsdialog geoeffnet. Bitte %d Punkte anklicken.", n_points
    )

    result = None
    while True:
        canvas = _draw_points(display.copy(), clicked)

        remaining = n_points - len(clicked)
        info = (
            f"Punkt {len(clicked)+1}/{n_points} anklicken"
            if remaining > 0
            else "Enter bestaetigen | R = zuruecksetzen | ESC abbrechen"
        )
        cv2.putText(
            canvas, info, (10, 30), _FONT, _FONT_SCALE, (0, 200, 255), _FONT_THICKNESS, cv2.LINE_AA
        )
        cv2.imshow(window_title, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            logger.info("Kalibrierung abgebrochen.")
            result = None
            break
        elif key in (13, 10) and len(clicked) >= 4:  # Enter
            if len(clicked) < n_points:
                logger.warning(
                    "Nur %d/%d Punkte gewaehlt, aber Bestaetigung akzeptiert.",
                    len(clicked), n_points,
                )
            orig_points = [
                (float(x / scale), float(y / scale)) for x, y in clicked
            ]
            logger.info("Kalibrierung bestaetigt mit %d Punkten.", len(orig_points))
            result = orig_points
            break
        elif key == ord("r"):
            clicked.clear()
            logger.debug("Punkte zurueckgesetzt.")

    cv2.destroyWindow(window_title)
    return result
