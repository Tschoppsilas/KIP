"""Kalibrierungs-Dialog: Spielfeld-Punkte im Video auf das Taktikboard mappen."""

import json
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.video_processing.homography import HomographyTransformer
from src.utils import get_logger

logger = get_logger(__name__)

# Farbpalette für die Punkt-Paare (bis zu 8 Paare)
PAIR_COLORS = [
    QColor(220, 50, 50),
    QColor(50, 180, 50),
    QColor(50, 100, 240),
    QColor(230, 180, 0),
    QColor(180, 0, 180),
    QColor(0, 180, 180),
    QColor(230, 110, 0),
    QColor(100, 0, 220),
]

MIN_POINTS = 4
MAX_POINTS = 8


# ---------------------------------------------------------------------------
# Orientierung: Hilfslinien + empfohlene Punktpositionen (Video / Board)
# ---------------------------------------------------------------------------

def _video_guide_positions(w: int, h: int) -> List[Tuple[float, float]]:
    """8 Positionen entlang eines typischen sichtbaren Spielfeld-Rechtecks (Randbereich)."""
    mx, my = int(0.08 * w), int(0.06 * h)
    x1, y1 = float(mx), float(my)
    x2, y2 = float(w - mx), float(h - my)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [
        (x1, y1), (cx, y1), (x2, y1), (x2, cy),
        (x2, y2), (cx, y2), (x1, y2), (x1, cy),
    ]


def _board_guide_positions(bw: int, bh: int) -> List[Tuple[float, float]]:
    """Passende 8 Zielpunkte auf dem Taktikboard (Ecken + Mittelpunkte der Kanten)."""
    return [
        (50.0, 50.0),
        (bw / 2, 50.0),
        (bw - 50.0, 50.0),
        (bw - 50.0, bh / 2),
        (bw - 50.0, bh - 50.0),
        (bw / 2, bh - 50.0),
        (50.0, bh - 50.0),
        (50.0, bh / 2),
    ]


def apply_video_field_guide(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Blendet schematische Feldhilfslinien und nummerierte Empfehlungspunkte ein.
    Keine echte Kantenerkennung — nur Orientierung für den Trainer.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    layer = np.zeros_like(out)
    mx, my = int(0.08 * w), int(0.06 * h)
    x1, y1 = mx, my
    x2, y2 = w - mx, h - my

    cv2.rectangle(layer, (x1, y1), (x2, y2), (0, 220, 180), 2)
    cx = (x1 + x2) // 2
    cv2.line(layer, (cx, y1), (cx, y2), (0, 200, 255), 2)
    gy1 = y1 + (y2 - y1) // 3
    gy2 = y1 + 2 * (y2 - y1) // 3
    cv2.line(layer, (x1, gy1), (x2, gy1), (120, 160, 255), 1)
    cv2.line(layer, (x1, gy2), (x2, gy2), (120, 160, 255), 1)

    alpha = 0.38
    cv2.addWeighted(layer, alpha, out, 1.0 - alpha, 0, out)

    hints = _video_guide_positions(w, h)
    for i, (px, py) in enumerate(hints):
        cv2.circle(out, (int(px), int(py)), 22, (40, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out, str(i + 1), (int(px) - 7, int(py) + 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            out, str(i + 1), (int(px) - 7, int(py) + 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 1, cv2.LINE_AA,
        )
    return out


def apply_board_field_guide(board_bgr: np.ndarray) -> np.ndarray:
    """Empfohlene Zielpunkte auf dem Board-Bild (gleiche Nummerierung wie Video)."""
    out = board_bgr.copy()
    bh, bw = out.shape[:2]
    hints = _board_guide_positions(bw, bh)
    for i, (px, py) in enumerate(hints):
        cv2.circle(out, (int(px), int(py)), 22, (255, 180, 60), 2, cv2.LINE_AA)
        cv2.putText(
            out, str(i + 1), (int(px) - 7, int(py) + 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            out, str(i + 1), (int(px) - 7, int(py) + 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 1, cv2.LINE_AA,
        )
    cv2.line(out, (bw // 2, 0), (bw // 2, bh), (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(out, (0, bh // 2), (bw, bh // 2), (120, 120, 120), 1, cv2.LINE_AA)
    return out


def numpy_bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class _ClickableImage(QLabel):
    """QLabel, das Mausklick-Koordinaten (relativ zum Originalbild) emittiert."""

    clicked = pyqtSignal(float, float)
    rightClicked = pyqtSignal(float, float)
    zoomFactorChanged = pyqtSignal(float)
    displayGeometryChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.WheelFocus)
        self._orig_w = 1
        self._orig_h = 1
        self._scroll_area: Optional[QScrollArea] = None
        self._zoom_scale = 1.0

    def attach_scroll_area(self, scroll: QScrollArea) -> None:
        self._scroll_area = scroll
        scroll.viewport().installEventFilter(self)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_zoom_factor(self, factor: float) -> None:
        self._zoom_scale = max(0.15, min(8.0, float(factor)))
        self._refresh_display()
        self.zoomFactorChanged.emit(self._zoom_scale)

    def eventFilter(self, obj: object, event: QEvent) -> bool:
        if (
            self._scroll_area is not None
            and obj is self._scroll_area.viewport()
            and event.type() == QEvent.Resize
        ):
            self._refresh_display()
        return False

    def _refresh_display(self) -> None:
        if not hasattr(self, "_base_pixmap"):
            return
        if self._scroll_area is None:
            disp = self._base_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(disp)
            return
        vp = self._scroll_area.viewport()
        vw = max(80, vp.width())
        vh = max(80, vp.height())
        fitted = self._base_pixmap.scaled(
            vw, vh, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        tw = max(1, int(fitted.width() * self._zoom_scale))
        th = max(1, int(fitted.height() * self._zoom_scale))
        disp = self._base_pixmap.scaled(
            tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setMinimumSize(disp.size())
        self.resize(disp.size())
        self.setPixmap(disp)
        if self._scroll_area is not None:
            self.displayGeometryChanged.emit()

    def set_image(self, img_bgr: np.ndarray) -> None:
        self._orig_h, self._orig_w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._base_pixmap = QPixmap.fromImage(qimg)
        self._refresh_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._scroll_area is None and hasattr(self, "_base_pixmap"):
            self._refresh_display()

    def wheelEvent(self, event):
        if self._scroll_area is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.set_zoom_factor(self._zoom_scale * 1.12)
        elif delta < 0:
            self.set_zoom_factor(self._zoom_scale / 1.12)
        event.accept()

    def _img_coords_from_event(self, event) -> Optional[Tuple[float, float]]:
        """Bildkoordinaten aus einem Mausereignis, oder None wenn außerhalb."""
        if not hasattr(self, "_base_pixmap"):
            return None
        disp = self._base_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        offset_x = (self.width() - disp.width()) / 2
        offset_y = (self.height() - disp.height()) / 2
        rel_x = event.x() - offset_x
        rel_y = event.y() - offset_y
        if rel_x < 0 or rel_y < 0 or rel_x > disp.width() or rel_y > disp.height():
            return None
        orig_x = rel_x / disp.width() * self._orig_w
        orig_y = rel_y / disp.height() * self._orig_h
        return orig_x, orig_y

    def mousePressEvent(self, event):
        coords = self._img_coords_from_event(event)
        if coords is None:
            return
        if event.button() == Qt.LeftButton:
            self.clicked.emit(*coords)
        elif event.button() == Qt.RightButton:
            self.rightClicked.emit(*coords)

    def draw_points(self, points: List[Tuple[float, float]], active_idx: Optional[int] = None) -> None:
        """Zeichnet nummerierte Punkte auf das Bild."""
        if not hasattr(self, "_base_pixmap"):
            return
        disp = self._base_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        canvas = QPixmap(self.size())
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        offset_x = (self.width() - disp.width()) / 2
        offset_y = (self.height() - disp.height()) / 2
        painter.drawPixmap(int(offset_x), int(offset_y), disp)

        scale_x = disp.width() / self._orig_w
        scale_y = disp.height() / self._orig_h

        for i, (px, py) in enumerate(points):
            color = PAIR_COLORS[i % len(PAIR_COLORS)]
            sx = int(px * scale_x + offset_x)
            sy = int(py * scale_y + offset_y)
            pen = QPen(color, 3)
            painter.setPen(pen)
            painter.setBrush(color)
            painter.drawEllipse(sx - 7, sy - 7, 14, 14)
            painter.setPen(QPen(Qt.white, 1))
            font = QFont("Arial", 8, QFont.Bold)
            painter.setFont(font)
            painter.drawText(sx - 4, sy + 5, str(i + 1))

        if active_idx is not None and active_idx < len(points):
            px, py = points[active_idx]
            sx = int(px * scale_x + offset_x)
            sy = int(py * scale_y + offset_y)
            pen = QPen(Qt.white, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(sx - 12, sy - 12, 24, 24)

        painter.end()
        self.setPixmap(canvas)


class CalibrationDialog(QDialog):
    """
    Kalibrierungs-Dialog für Homography.

    Workflow:
    1. Video zeigt Hilfslinien + empfohlene Punkte 1–8; Board ebenfalls 1–8.
    2. Klick auf Video → Klick auf passenden Punkt auf dem Board (Paar).
    3. Ab 4 Paaren: Live-Vorschau der Homography; Speichern wenn zufrieden.
    """

    def __init__(
        self,
        video_frame: np.ndarray,
        board_img: np.ndarray,
        save_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.Window | Qt.WindowMinMaxButtonsHint)
        self.setWindowTitle("Kalibrierung – Spielfeld-Punkte setzen")
        self.setMinimumSize(1280, 720)

        self._frame_raw = video_frame.copy()
        self._board_raw = board_img.copy()
        self._save_path = save_path

        self._video_display_bgr = apply_video_field_guide(self._frame_raw)
        self._board_display_bgr = apply_board_field_guide(self._board_raw)

        self._src_points: List[Tuple[float, float]] = []
        self._dst_points: List[Tuple[float, float]] = []
        self._waiting_for_board = False

        self._in_preview_update = False

        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        info = QLabel(
            "<b>Orientierung:</b> Orange/grüne Hilfslinien und Kreise <b>1–8</b> sind nur Vorschläge — "
            "klicke die <i>tatsächlichen</i> Spielfeldmarkierungen im Video und die passenden Punkte auf dem Board. "
            "Ab 4 Paaren siehst du rechts eine <b>Vorschau</b> (Video auf Board projiziert). "
            "<b>Rechtsklick auf einen gesetzten Punkt</b> löscht das zugehörige Paar. "
            "<b>Fenster maximieren</b> und <b>Zoom</b> (Schieberegler oder Mausrad über Video/Board)."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "font-size: 12px; padding: 8px; background: #1e1e2e; color: #cdd6f4; border-radius: 4px;"
        )
        root.addWidget(info)

        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 6px; "
            "background: #1e1e2e; color: #cdd6f4; border-radius: 4px;"
        )
        root.addWidget(self._status_label)

        img_row = QHBoxLayout()

        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Video</b> — Hilfslinien + Vorschlag 1–8, dann deine Klicks"))
        self._video_scroll = QScrollArea()
        self._video_scroll.setWidgetResizable(False)
        self._video_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._video_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._video_scroll.setMinimumHeight(280)
        self._video_scroll.setStyleSheet("border: 1px solid #444; background: #222;")
        self._video_lbl = _ClickableImage()
        self._video_lbl.attach_scroll_area(self._video_scroll)
        self._video_scroll.setWidget(self._video_lbl)
        self._video_lbl.clicked.connect(self._on_video_click)
        self._video_lbl.rightClicked.connect(self._on_video_right_click)
        self._video_zoom = QSlider(Qt.Horizontal)
        self._video_zoom.setRange(25, 400)
        self._video_zoom.setValue(100)
        self._video_zoom.valueChanged.connect(
            lambda v: self._video_lbl.set_zoom_factor(v / 100.0)
        )
        self._video_lbl.zoomFactorChanged.connect(self._paint_overlays_only)
        self._video_zoom_pct = QLabel("100 %")
        self._video_zoom_pct.setMinimumWidth(48)
        self._video_lbl.zoomFactorChanged.connect(
            lambda z: self._video_zoom_pct.setText(f"{int(round(z * 100))} %")
        )
        self._video_lbl.zoomFactorChanged.connect(
            lambda z: self._sync_zoom_slider(self._video_zoom, z)
        )
        self._video_lbl.displayGeometryChanged.connect(self._paint_overlays_only)
        video_zoom_row = QHBoxLayout()
        video_zoom_row.addWidget(QLabel("Zoom:"))
        video_zoom_row.addWidget(self._video_zoom, 1)
        video_zoom_row.addWidget(self._video_zoom_pct)
        left.addWidget(self._video_scroll)
        left.addLayout(video_zoom_row)

        mid = QVBoxLayout()
        mid.addWidget(QLabel("<b>Taktikboard</b> — gleiche Nummern als Ziel, dann klicken"))
        self._board_scroll = QScrollArea()
        self._board_scroll.setWidgetResizable(False)
        self._board_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._board_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._board_scroll.setMinimumHeight(280)
        self._board_scroll.setStyleSheet("border: 1px solid #444; background: #222;")
        self._board_lbl = _ClickableImage()
        self._board_lbl.attach_scroll_area(self._board_scroll)
        self._board_scroll.setWidget(self._board_lbl)
        self._board_lbl.clicked.connect(self._on_board_click)
        self._board_lbl.rightClicked.connect(self._on_board_right_click)
        self._board_zoom = QSlider(Qt.Horizontal)
        self._board_zoom.setRange(25, 400)
        self._board_zoom.setValue(100)
        self._board_zoom.valueChanged.connect(
            lambda v: self._board_lbl.set_zoom_factor(v / 100.0)
        )
        self._board_lbl.zoomFactorChanged.connect(self._paint_overlays_only)
        self._board_zoom_pct = QLabel("100 %")
        self._board_zoom_pct.setMinimumWidth(48)
        self._board_lbl.zoomFactorChanged.connect(
            lambda z: self._board_zoom_pct.setText(f"{int(round(z * 100))} %")
        )
        self._board_lbl.zoomFactorChanged.connect(
            lambda z: self._sync_zoom_slider(self._board_zoom, z)
        )
        self._board_lbl.displayGeometryChanged.connect(self._paint_overlays_only)
        board_zoom_row = QHBoxLayout()
        board_zoom_row.addWidget(QLabel("Zoom:"))
        board_zoom_row.addWidget(self._board_zoom, 1)
        board_zoom_row.addWidget(self._board_zoom_pct)
        mid.addWidget(self._board_scroll)
        mid.addLayout(board_zoom_row)

        prev = QVBoxLayout()
        prev.addWidget(QLabel("<b>Vorschau Homography</b> (Video → Board, überblendet)"))
        self._preview_lbl = QLabel()
        self._preview_lbl.setAlignment(Qt.AlignCenter)
        self._preview_lbl.setMinimumSize(320, 200)
        self._preview_lbl.setStyleSheet("background: #11111a; color: #888; padding: 8px;")
        self._preview_lbl.setWordWrap(True)
        self._preview_lbl.setText(
            "Sobald mindestens 4 vollständige Punkt-Paare gesetzt sind,\n"
            "erscheint hier eine Live-Vorschau."
        )
        prev.addWidget(self._preview_lbl)

        img_row.addLayout(left, 2)
        img_row.addLayout(mid, 2)
        img_row.addLayout(prev, 2)

        root.addLayout(img_row)

        btn_row = QHBoxLayout()
        self._undo_btn = QPushButton("Letzten Punkt entfernen")
        self._undo_btn.clicked.connect(self._undo_last)
        btn_row.addWidget(self._undo_btn)

        self._load_btn = QPushButton("Kalibrierung laden …")
        self._load_btn.clicked.connect(self._load_calibration)
        btn_row.addWidget(self._load_btn)

        btn_row.addStretch()

        self._save_btn = QPushButton("Berechnen & Speichern")
        self._save_btn.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self._save_btn.clicked.connect(self._save_calibration)
        btn_row.addWidget(self._save_btn)

        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        root.addLayout(btn_row)

    def _sync_zoom_slider(self, slider: QSlider, z: float) -> None:
        val = int(round(z * 100))
        val = max(slider.minimum(), min(slider.maximum(), val))
        if slider.value() == val:
            return
        slider.blockSignals(True)
        slider.setValue(val)
        slider.blockSignals(False)

    def _paint_overlays_only(self) -> None:
        """Nur Punkte zeichnen — kein set_image (vermeidet Rekursion über displayGeometryChanged)."""
        active = len(self._dst_points) if self._waiting_for_board else None
        self._video_lbl.draw_points(self._src_points, active_idx=active)
        self._board_lbl.draw_points(self._dst_points)

    # ------------------------------------------------------------------
    # Klick-Handler
    # ------------------------------------------------------------------
    def _on_video_click(self, x: float, y: float) -> None:
        if self._waiting_for_board:
            return
        if len(self._src_points) >= MAX_POINTS:
            QMessageBox.information(self, "Maximum erreicht", f"Maximal {MAX_POINTS} Punkte erlaubt.")
            return
        self._src_points.append((x, y))
        self._waiting_for_board = True
        self._refresh()

    def _on_board_click(self, x: float, y: float) -> None:
        if not self._waiting_for_board:
            return
        self._dst_points.append((x, y))
        self._waiting_for_board = False
        self._refresh()

    def _undo_last(self) -> None:
        if self._waiting_for_board:
            self._src_points.pop()
            self._waiting_for_board = False
        elif self._dst_points:
            self._src_points.pop()
            self._dst_points.pop()
        self._refresh()

    def _hit_index(
        self, points: List[Tuple[float, float]], x: float, y: float, img_w: int, img_h: int
    ) -> Optional[int]:
        """Index des nächsten Punktes innerhalb des Trefferradius, oder None."""
        radius = max(img_w, img_h) * 0.025
        best_dist = radius
        best_idx = None
        for i, (px, py) in enumerate(points):
            dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _delete_pair(self, idx: int) -> None:
        """Löscht das Punkt-Paar bei Index idx (beide Listen + ausstehenden Zustand)."""
        n_complete = len(self._dst_points)
        if idx < n_complete:
            self._src_points.pop(idx)
            self._dst_points.pop(idx)
            if self._waiting_for_board:
                # ausstehender Punkt rutscht auf den vorigen Slot
                pass
        elif idx == n_complete and self._waiting_for_board:
            # ausstehender src-Punkt ohne Board-Pendant
            self._src_points.pop(idx)
            self._waiting_for_board = False
        self._refresh()

    def _on_video_right_click(self, x: float, y: float) -> None:
        idx = self._hit_index(
            self._src_points, x, y,
            self._video_lbl._orig_w, self._video_lbl._orig_h,
        )
        if idx is not None:
            self._delete_pair(idx)

    def _on_board_right_click(self, x: float, y: float) -> None:
        idx = self._hit_index(
            self._dst_points, x, y,
            self._board_lbl._orig_w, self._board_lbl._orig_h,
        )
        if idx is not None:
            self._delete_pair(idx)

    # ------------------------------------------------------------------
    # Vorschau
    # ------------------------------------------------------------------
    def _update_preview(self) -> None:
        if self._in_preview_update:
            return
        self._in_preview_update = True
        try:
            self._update_preview_impl()
        finally:
            self._in_preview_update = False

    def _update_preview_impl(self) -> None:
        pairs = len(self._dst_points)
        if pairs < MIN_POINTS:
            self._preview_lbl.clear()
            self._preview_lbl.setText(
                "Sobald mindestens 4 vollständige Punkt-Paare gesetzt sind,\n"
                "erscheint hier eine Live-Vorschau."
            )
            self._preview_lbl.setStyleSheet("background: #11111a; color: #888; padding: 8px;")
            return

        src = np.array(self._src_points[:pairs], dtype=np.float32)
        dst = np.array(self._dst_points[:pairs], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            self._preview_lbl.setText("Homography konnte nicht berechnet werden.")
            return

        bh, bw = self._board_raw.shape[:2]
        warped = cv2.warpPerspective(self._frame_raw, H, (bw, bh))
        blend = cv2.addWeighted(warped, 0.45, self._board_raw, 0.55, 0)

        pix = numpy_bgr_to_pixmap(blend)
        scaled = pix.scaled(
            self._preview_lbl.width() if self._preview_lbl.width() > 50 else 400,
            self._preview_lbl.height() if self._preview_lbl.height() > 50 else 300,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._preview_lbl.setPixmap(scaled)
        self._preview_lbl.setStyleSheet("background: #11111a; padding: 4px;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview()

    # ------------------------------------------------------------------
    # Speichern / Laden
    # ------------------------------------------------------------------
    def _save_calibration(self) -> None:
        if len(self._src_points) < MIN_POINTS or len(self._src_points) != len(self._dst_points):
            QMessageBox.warning(self, "Zu wenige Punkte", f"Mindestens {MIN_POINTS} vollständige Punkt-Paare erforderlich.")
            return
        try:
            transformer = HomographyTransformer()
            transformer.compute(self._src_points, self._dst_points)
            transformer.save(self._save_path)
            self._transformer = transformer
            QMessageBox.information(
                self, "Gespeichert",
                f"Kalibrierung mit {len(self._src_points)} Paaren gespeichert:\n{self._save_path}"
            )
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Fehler", str(exc))

    def _load_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Kalibrierung laden", "", "JSON-Dateien (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._src_points = [tuple(p) for p in data["src_points"]]
            self._dst_points = [tuple(p) for p in data["dst_points"]]
            self._waiting_for_board = False
            self._save_path = path
            self._refresh()
            logger.info(f"Kalibrierung geladen: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Fehler beim Laden", str(exc))

    # ------------------------------------------------------------------
    # UI-Aktualisierung
    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        pairs = len(self._dst_points)

        if self._waiting_for_board:
            status = f"Punkt {pairs + 1}: Jetzt den zugehörigen Punkt auf dem Taktikboard klicken."
            status_color = "#fab387"
            active = pairs
        elif pairs == 0:
            status = "Schritt 1: Klicke einen Punkt im Video (Orientierung: Kreise 1–8 und Hilfslinien)."
            status_color = "#89dceb"
            active = None
        elif pairs < MIN_POINTS:
            status = f"{pairs} Paare — noch {MIN_POINTS - pairs} bis zur Vorschau. Nächsten Video-Punkt klicken."
            status_color = "#a6e3a1"
            active = None
        else:
            status = (
                f"{pairs} Punkt-Paare. Vorschau aktiv — bei Bedarf mehr Punkte (max. {MAX_POINTS}) "
                "oder „Berechnen & Speichern“."
            )
            status_color = "#a6e3a1"
            active = None

        self._status_label.setText(status)
        self._status_label.setStyleSheet(
            f"font-size: 13px; font-weight: bold; padding: 6px; "
            f"background: #1e1e2e; color: {status_color}; border-radius: 4px;"
        )

        self._video_lbl.set_image(self._video_display_bgr)
        self._video_lbl.draw_points(self._src_points, active_idx=active)

        self._board_lbl.set_image(self._board_display_bgr)
        self._board_lbl.draw_points(self._dst_points)

        self._update_preview()

        can_save = pairs >= MIN_POINTS and not self._waiting_for_board
        self._save_btn.setEnabled(can_save)
        self._undo_btn.setEnabled(len(self._src_points) > 0)

    def showEvent(self, event):
        super().showEvent(event)
        self._update_preview()

    # ------------------------------------------------------------------
    # Ergebnis nach Schliessen
    # ------------------------------------------------------------------
    def get_transformer(self) -> Optional[HomographyTransformer]:
        return getattr(self, "_transformer", None)
