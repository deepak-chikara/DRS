"""Calibration tests."""

from pathlib import Path

from drs.fusion.calibration import PitchCalibration, StumpPoints, is_inside_corridor


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
