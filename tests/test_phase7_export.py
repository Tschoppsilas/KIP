"""Phase 7 Tests: Datenexport – PNG, PDF, Video, Temp-Preview, ExportPreset.

Testet:
- export_png: Datei wird erstellt, ist lesbar, hat korrekte Grösse
- export_pdf: Datei wird erstellt, ist eine gültige PDF
- export_image: Routing nach Format (png/pdf)
- export_video: Datei wird erstellt, korrekte Frame-Anzahl
- export_video_from_states: Integration mit BoardRenderer + BoardState
- preview_temp: Kein Fehler, keine persistente Projektdatei
- ExportPreset: Felder und Vorgaben
- Definition of Done: Exportierte Dateien sind öffenbar und enthalten Overlays
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


BOARD_IMG = Path(__file__).parent.parent / "Taktikboard" / "Taktikboard.png"


def _dummy_image(w=200, h=100, color=(200, 80, 40)) -> Image.Image:
    img = Image.new("RGB", (w, h), color)
    return img


def _dummy_rgba(w=200, h=100) -> Image.Image:
    img = Image.new("RGBA", (w, h), (100, 150, 200, 128))
    return img


# ---------------------------------------------------------------------------
# ExportPreset
# ---------------------------------------------------------------------------

class TestExportPreset(unittest.TestCase):

    def test_default_preset_format(self):
        from src.utils.exporter import PRESET_DEFAULT
        self.assertEqual(PRESET_DEFAULT.format, "png")

    def test_pdf_preset_format(self):
        from src.utils.exporter import PRESET_PDF
        self.assertEqual(PRESET_PDF.format, "pdf")

    def test_video_preset_fps(self):
        from src.utils.exporter import PRESET_VIDEO_HD
        self.assertAlmostEqual(PRESET_VIDEO_HD.video_fps, 30.0)

    def test_custom_preset(self):
        from src.utils.exporter import ExportPreset
        p = ExportPreset(format="png", png_compress=0, pdf_dpi=300)
        self.assertEqual(p.png_compress, 0)
        self.assertEqual(p.pdf_dpi, 300)


# ---------------------------------------------------------------------------
# PNG-Export
# ---------------------------------------------------------------------------

class TestExportPng(unittest.TestCase):

    def test_creates_file(self):
        from src.utils.exporter import export_png
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "board.png"
            result = export_png(_dummy_image(), out)
            self.assertTrue(result.exists())

    def test_file_is_readable_image(self):
        from src.utils.exporter import export_png
        with tempfile.TemporaryDirectory() as tmp:
            out = export_png(_dummy_image(300, 150), Path(tmp) / "test.png")
            loaded = Image.open(out)
            self.assertEqual(loaded.size, (300, 150))

    def test_forces_png_extension(self):
        from src.utils.exporter import export_png
        with tempfile.TemporaryDirectory() as tmp:
            out = export_png(_dummy_image(), Path(tmp) / "board.jpg")
            self.assertEqual(out.suffix, ".png")

    def test_rgba_input_converted(self):
        """RGBA-Eingabe muss verlustfrei als RGB gespeichert werden."""
        from src.utils.exporter import export_png
        with tempfile.TemporaryDirectory() as tmp:
            out = export_png(_dummy_rgba(), Path(tmp) / "rgba.png")
            loaded = Image.open(out)
            self.assertEqual(loaded.mode, "RGB")

    def test_creates_parent_dirs(self):
        from src.utils.exporter import export_png
        with tempfile.TemporaryDirectory() as tmp:
            out = export_png(_dummy_image(), Path(tmp) / "sub" / "dir" / "board.png")
            self.assertTrue(out.exists())


# ---------------------------------------------------------------------------
# PDF-Export
# ---------------------------------------------------------------------------

class TestExportPdf(unittest.TestCase):

    def test_creates_file(self):
        from src.utils.exporter import export_pdf
        with tempfile.TemporaryDirectory() as tmp:
            out = export_pdf(_dummy_image(), Path(tmp) / "board.pdf")
            self.assertTrue(out.exists())

    def test_file_has_pdf_magic_bytes(self):
        from src.utils.exporter import export_pdf
        with tempfile.TemporaryDirectory() as tmp:
            out = export_pdf(_dummy_image(), Path(tmp) / "board.pdf")
            magic = out.read_bytes()[:4]
            self.assertEqual(magic, b"%PDF")

    def test_forces_pdf_extension(self):
        from src.utils.exporter import export_pdf
        with tempfile.TemporaryDirectory() as tmp:
            out = export_pdf(_dummy_image(), Path(tmp) / "board.png")
            self.assertEqual(out.suffix, ".pdf")


# ---------------------------------------------------------------------------
# export_image (Routing)
# ---------------------------------------------------------------------------

class TestExportImage(unittest.TestCase):

    def test_routes_to_png_by_default(self):
        from src.utils.exporter import export_image, PRESET_DEFAULT
        with tempfile.TemporaryDirectory() as tmp:
            out = export_image(_dummy_image(), Path(tmp) / "out.png", PRESET_DEFAULT)
            self.assertEqual(out.suffix, ".png")

    def test_routes_to_pdf_by_preset(self):
        from src.utils.exporter import export_image, PRESET_PDF
        with tempfile.TemporaryDirectory() as tmp:
            out = export_image(_dummy_image(), Path(tmp) / "out.pdf", PRESET_PDF)
            self.assertEqual(out.suffix, ".pdf")

    def test_routes_to_pdf_by_extension(self):
        from src.utils.exporter import export_image
        with tempfile.TemporaryDirectory() as tmp:
            out = export_image(_dummy_image(), Path(tmp) / "doc.pdf")
            self.assertEqual(out.suffix, ".pdf")


# ---------------------------------------------------------------------------
# Video-Export
# ---------------------------------------------------------------------------

class TestExportVideo(unittest.TestCase):

    def _make_frames(self, n=5, w=320, h=240):
        return [_dummy_image(w, h) for _ in range(n)]

    def test_creates_file(self):
        from src.utils.exporter import export_video
        with tempfile.TemporaryDirectory() as tmp:
            out = export_video(self._make_frames(), Path(tmp) / "out.mp4", fps=10.0)
            self.assertTrue(out.exists())

    def test_file_is_valid_video(self):
        import cv2
        from src.utils.exporter import export_video
        with tempfile.TemporaryDirectory() as tmp:
            out = export_video(self._make_frames(10), Path(tmp) / "out.mp4", fps=5.0)
            cap = cv2.VideoCapture(str(out))
            self.assertTrue(cap.isOpened())
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.assertGreaterEqual(frame_count, 9)

    def test_numpy_frames_accepted(self):
        from src.utils.exporter import export_video
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            out = export_video(frames, Path(tmp) / "np.mp4", fps=5.0)
            self.assertTrue(out.exists())

    def test_empty_frames_raises(self):
        from src.utils.exporter import export_video
        with self.assertRaises(ValueError):
            export_video([], "/tmp/empty.mp4")

    def test_forces_mp4_extension(self):
        from src.utils.exporter import export_video
        with tempfile.TemporaryDirectory() as tmp:
            out = export_video(self._make_frames(2), Path(tmp) / "out.avi", fps=5.0)
            self.assertEqual(out.suffix, ".mp4")


# ---------------------------------------------------------------------------
# export_video_from_states (BoardRenderer Integration)
# ---------------------------------------------------------------------------

class TestExportVideoFromStates(unittest.TestCase):

    def test_renders_states_to_video(self):
        from src.utils.exporter import export_video_from_states, PRESET_VIDEO_HD
        from src.gui.board_renderer import BoardRenderer, BoardState, PlayerSymbol

        renderer = BoardRenderer(BOARD_IMG)
        states = [
            BoardState(players=[
                PlayerSymbol(i + 1, team=i % 2, class_name="player",
                             board_x=float(i * 5), board_y=10.0)
            ], frame_index=i)
            for i in range(4)
        ]
        preset = PRESET_VIDEO_HD
        preset.video_fps = 5.0

        with tempfile.TemporaryDirectory() as tmp:
            out = export_video_from_states(states, renderer,
                                           Path(tmp) / "states.mp4", preset)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)


# ---------------------------------------------------------------------------
# Should: preview_temp (kein Fehler, kein Projektverzeichnis-Artefakt)
# ---------------------------------------------------------------------------

class TestPreviewTemp(unittest.TestCase):

    def test_does_not_raise(self):
        from src.utils.exporter import preview_temp
        # open_viewer=False → kein xdg-open im Test
        preview_temp(_dummy_image(), open_viewer=False)

    def test_no_file_in_project_dir(self):
        """Temporäre Datei darf nicht im Projektverzeichnis landen."""
        import os
        from src.utils.exporter import preview_temp
        project_root = Path(__file__).parent.parent
        before = set(project_root.glob("univision_preview_*.png"))
        preview_temp(_dummy_image(), open_viewer=False)
        after = set(project_root.glob("univision_preview_*.png"))
        self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# Definition of Done: Exportdateien sind öffenbar und enthalten Overlays
# ---------------------------------------------------------------------------

class TestDefinitionOfDone(unittest.TestCase):

    def test_png_with_overlay_readable(self):
        """Board mit Spieler-Overlay als PNG exportiert und wieder einlesbar."""
        from src.utils.exporter import export_png
        from src.gui.board_renderer import BoardRenderer, BoardState, PlayerSymbol, Arrow

        renderer = BoardRenderer(BOARD_IMG)
        state = BoardState(
            players=[PlayerSymbol(1, team=0, class_name="player",
                                  board_x=20.0, board_y=10.0)],
            arrows=[Arrow(5.0, 5.0, 25.0, 10.0, kind="pass")],
            frame_index=0,
        )
        img = renderer.render_rgb(state)
        with tempfile.TemporaryDirectory() as tmp:
            out = export_png(img, Path(tmp) / "done.png")
            loaded = Image.open(out)
            self.assertEqual(loaded.size, (1280, 720))

    def test_pdf_with_overlay_has_correct_magic(self):
        """Board mit Overlay als PDF – Datei hat korrekte PDF-Signatur."""
        from src.utils.exporter import export_pdf
        from src.gui.board_renderer import BoardRenderer, BoardState, PlayerSymbol

        renderer = BoardRenderer(BOARD_IMG)
        state = BoardState(players=[PlayerSymbol(2, team=1, class_name="goalkeeper",
                                                  board_x=2.0, board_y=10.0)])
        img = renderer.render_rgb(state)
        with tempfile.TemporaryDirectory() as tmp:
            out = export_pdf(img, Path(tmp) / "done.pdf")
            self.assertEqual(out.read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
