from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - environment-dependent
    cv2 = None
    np = None

if cv2 is not None and np is not None:
    from video_processing.video_reader import read_first_frame
else:  # pragma: no cover - environment-dependent
    read_first_frame = None


@unittest.skipUnless(cv2 is not None and np is not None, "opencv/numpy nicht installiert")
class TestVideoFrameLoading(unittest.TestCase):
    def test_read_first_frame_loads_valid_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "dummy.avi"
            width, height = 64, 48

            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (width, height),
            )
            self.assertTrue(writer.isOpened(), "VideoWriter konnte nicht geoeffnet werden.")

            first_frame = np.zeros((height, width, 3), dtype=np.uint8)
            first_frame[:, :] = (0, 255, 0)
            writer.write(first_frame)
            writer.release()

            loaded_frame = read_first_frame(video_path)

            self.assertIsNotNone(loaded_frame)
            self.assertEqual(loaded_frame.shape, (height, width, 3))
            self.assertEqual(loaded_frame.dtype, np.uint8)
