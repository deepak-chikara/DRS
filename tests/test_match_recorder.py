"""Match recorder tests."""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from drs.recording.match_recorder import MatchRecorder


def test_match_recorder_writes_frames():
    with tempfile.TemporaryDirectory() as tmp:
        rec = MatchRecorder("test", fps=10.0, output_dir=Path(tmp), segment_minutes=60)
        rec.start()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(5):
            rec.write(frame)
        path = rec.stop()
        assert path is not None
        assert path.is_file()
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened()
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        assert count == 5
