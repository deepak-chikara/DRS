"""Stump line locking tests."""

import cv2
import numpy as np

from drs.config import DRSConfig
from drs.fusion.calibration import corridor_bounds_at_y, stump_x_at_y
from drs.ui.stumps import stump_points_from_pitch_contour


def _fake_pitch_contour(w: int, h: int, offset_x: int = 100, offset_y: int = 50):
    pts = np.array([
        [[offset_x, offset_y]],
        [[offset_x + w, offset_y]],
        [[offset_x + w, offset_y + h]],
        [[offset_x, offset_y + h]],
    ], dtype=np.int32)
    return [pts]


def test_stump_points_from_pitch_contour():
    cfg = DRSConfig(stump_width_ratio=0.05, pitch_area_min=1000)
    contours = _fake_pitch_contour(400, 300)
    sp = stump_points_from_pitch_contour(contours, cfg, frame_h=720, frame_w=1280)
    assert sp is not None
    assert sp.bowler_off[1] < sp.striker_off[1]
    assert sp.bowler_off[0] < sp.bowler_leg[0]
    assert sp.striker_off[0] < sp.striker_leg[0]


def test_locked_corridor_width_stable_when_pitch_bbox_changes():
    cfg = DRSConfig(pitch_area_min=1000, stump_width_ratio=0.05)
    sp_locked = stump_points_from_pitch_contour(
        _fake_pitch_contour(400, 300), cfg, 720, 1280,
    )
    assert sp_locked is not None

    y = 200
    locked_width = corridor_bounds_at_y(sp_locked, y)[1] - corridor_bounds_at_y(sp_locked, y)[0]

    sp_wider = stump_points_from_pitch_contour(
        _fake_pitch_contour(600, 300), cfg, 720, 1280,
    )
    wider_width = corridor_bounds_at_y(sp_wider, y)[1] - corridor_bounds_at_y(sp_wider, y)[0]
    assert locked_width != wider_width
    assert corridor_bounds_at_y(sp_locked, y)[1] - corridor_bounds_at_y(sp_locked, y)[0] == locked_width


def test_draw_stump_corridor_extends_full_frame_height():
    cfg = DRSConfig(pitch_area_min=1000, stump_width_ratio=0.05)
    sp = stump_points_from_pitch_contour(_fake_pitch_contour(400, 300), cfg, 720, 1280)
    assert sp is not None
    # Lines extrapolated to frame top/bottom should differ from short pitch-only segment.
    short_span = abs(sp.striker_off[0] - sp.bowler_off[0])
    top_x = stump_x_at_y(sp.striker_off, sp.bowler_off, 0)
    bot_x = stump_x_at_y(sp.striker_off, sp.bowler_off, 719)
    assert top_x != bot_x or sp.bowler_off[0] == sp.striker_off[0]


def test_engine_locks_stumps_once():
    from drs.engine import DRSEngine

    cfg = DRSConfig(
        pitch_area_min=1000,
        stump_width_ratio=0.05,
        ground_id="__test_no_cal__",
        calibration_file="",
    )
    engine = DRSEngine(cfg)
    engine._stump_points = None
    engine._stump_points_locked = False
    engine._stump_from_calibration = False

    engine._try_lock_stumps_from_pitch(_fake_pitch_contour(400, 300), 720, 1280)
    first = engine.stump_points
    assert first is not None
    assert engine._stump_points_locked

    engine._try_lock_stumps_from_pitch(_fake_pitch_contour(800, 300), 720, 1280)
    assert engine.stump_points == first
