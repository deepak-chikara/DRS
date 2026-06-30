"""Golden-frame regression tests using synthetic cricket frames."""

import numpy as np

from drs.config import DRSConfig
from drs.engine import DRSEngine
from drs.fusion.calibration import StumpPoints, pixel_to_pitch_normalized
from drs.ui.pitch_diagram import render_pitch_diagram
from synthetic_frames import make_synthetic_frame


def _test_config() -> DRSConfig:
    return DRSConfig(
        mode="file",
        video_path="",
        detection_mode="color",
        detection_scale=1.0,
    )


def test_synthetic_frame_processes_without_error():
    cfg = _test_config()
    engine = DRSEngine(cfg)
    frame = make_synthetic_frame()
    out = engine.process_frame(frame)
    assert out is not None
    assert out.shape == frame.shape


def test_stump_mapper_center_line_on_wicket():
    stumps = StumpPoints(
        striker_off=(280, 200),
        striker_leg=(360, 200),
        bowler_off=(280, 320),
        bowler_leg=(360, 320),
    )
    nx, ny = pixel_to_pitch_normalized(
        320, 260,
        frame_w=640,
        frame_h=360,
        homography=None,
        stump_points=stumps,
    )
    assert 0.35 <= nx <= 0.65, f"expected near wicket line, got {nx}"
    assert 0.0 <= ny <= 1.0


def test_pitch_diagram_renders_for_synthetic_trajectory():
    points = [(0.5, 0.2), (0.5, 0.5), (0.5, 0.75)]
    img = render_pitch_diagram(points, pitch_bounce=(0.5, 0.5), impact=(0.5, 0.75))
    assert img is not None
    assert img.size > 0
    assert np.mean(img) > 0
