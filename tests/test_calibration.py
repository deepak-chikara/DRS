"""Calibration tests."""

from pathlib import Path

import numpy as np

from drs.fusion.calibration import (
    PitchCalibration,
    StumpPoints,
    STUMP_SET_WIDTH_FRAC,
    is_inside_corridor,
    is_usable_homography,
    pixel_to_pitch_from_stumps,
    pixel_to_pitch_normalized,
)


def test_stump_points_roundtrip():
    sp = StumpPoints((1, 2), (3, 4), (5, 6), (7, 8))
    d = sp.to_dict()
    loaded = StumpPoints.from_dict(d)
    assert loaded == sp


def test_is_inside_corridor():
    sp = StumpPoints((100, 500), (200, 500), (110, 100), (190, 100))
    assert is_inside_corridor(150, 300, sp)
    assert not is_inside_corridor(50, 300, sp)


def test_pitch_calibration_save_load(tmp_path: Path):
    sp = StumpPoints((10, 20), (30, 40), (50, 60), (70, 80))
    from drs.fusion.calibration import CameraCalibration
    import numpy as np

    cam = CameraCalibration("primary", np.eye(3), [], [], sp)
    cal = PitchCalibration("test_ground", 20.12, 3.05, {"primary": cam})
    out = tmp_path / "cal.json"
    cal.save(out)
    loaded = PitchCalibration.load(out)
    assert loaded is not None
    assert loaded.ground_id == "test_ground"
    pts = loaded.get_stump_points("primary")
    assert pts is not None
    assert pts.striker_off == (10, 20)


def _sample_stumps() -> StumpPoints:
    return StumpPoints(
        striker_off=(615, 221),
        striker_leg=(640, 221),
        bowler_off=(611, 495),
        bowler_leg=(643, 492),
    )


def test_is_usable_homography_rejects_identity():
    assert not is_usable_homography(np.eye(3))
    assert not is_usable_homography(None)


def test_pixel_to_pitch_from_stumps_center_mid_pitch():
    stumps = _sample_stumps()
    y = 358
    x_off, x_leg = 613.0, 641.5
    x_center = (x_off + x_leg) / 2
    nx, ny = pixel_to_pitch_from_stumps(int(x_center), y, stumps)
    assert abs(nx - 0.5) < 0.02
    assert abs(ny - 0.5) < 0.05


def test_pixel_to_pitch_from_stumps_off_stump_line():
    stumps = _sample_stumps()
    y = 358
    nx, _ = pixel_to_pitch_from_stumps(613, y, stumps)
    expected = 0.5 - STUMP_SET_WIDTH_FRAC / 2
    assert abs(nx - expected) < 0.02


def test_pixel_to_pitch_normalized_prefers_stumps_over_identity_homography():
    stumps = _sample_stumps()
    nx, ny = pixel_to_pitch_normalized(
        627, 376,
        frame_w=1280,
        frame_h=720,
        homography=np.eye(3),
        stump_points=stumps,
    )
    fallback_ny = 1.0 - 376 / 720
    stump_ny = (376 - 221) / (495 - 221)
    assert abs(ny - stump_ny) < 0.05
    assert abs(ny - fallback_ny) > 0.05
    assert 0.0 <= nx <= 1.0
