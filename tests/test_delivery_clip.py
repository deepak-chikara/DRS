"""Delivery clip export tests."""

import tempfile
import time
from pathlib import Path

import numpy as np

from drs.recording.delivery_clip import DeliveryClipExporter, export_drs_call_clip, slice_buffer_around_time
from drs.sync.frame_sync import BufferedFrame


def _frame(ts: float) -> BufferedFrame:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    return BufferedFrame(timestamp=ts, primary_frame=img, secondary_frame=None, state_snapshot={}, combined_frame=img)


def test_slice_buffer_around_time():
    t0 = time.time()
    buf = [_frame(t0 + i) for i in range(30)]
    sliced = slice_buffer_around_time(buf, t0 + 15, pre_seconds=5, post_seconds=5)
    assert len(sliced) == 11


def test_export_drs_call_clip():
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        buf = [_frame(t0 + i * 0.1) for i in range(200)]
        path = export_drs_call_clip(buf, t0 + 10, delivery_id=1, fps=10.0, exporter=DeliveryClipExporter(Path(tmp)))
        assert path is not None
        assert path.is_file()
