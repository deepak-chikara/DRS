"""Event detection tests."""

from drs.config import DRSConfig
from drs.decision.events import EventDetector
from drs.state import DRSState


def _moving_ball_state(y: int, y_prev: int, x: int = 500) -> DRSState:
    state = DRSState()
    state.ball.x = x
    state.ball.y = y
    state.ball.x_prev = x
    state.ball.y_prev = y_prev
    state.ball.prev_x_diff = 0
    state.ball.prev_y_diff = y - y_prev
    return state


def test_pitch_not_set_without_delivery_motion():
    cfg = DRSConfig(delivery_motion_min_px=4, delivery_motion_frames=2, pitch_stable_frames=2)
    detector = EventDetector(cfg)
    state = DRSState()
    state.ball.x = 500
    state.ball.y = 300
    state.ball.x_prev = 500
    state.ball.y_prev = 300
    state.ball.prev_x_diff = 0
    state.ball.prev_y_diff = 0

    pitch_cnt = [[[400, 250]], [[700, 250]], [[700, 500]], [[400, 500]]]
    for _ in range(5):
        detector.process_frame(state, [__import__("numpy").array(pitch_cnt)], [])

    assert state.pitch_point is None


def test_pitch_set_after_downward_motion_on_pitch():
    import numpy as np

    cfg = DRSConfig(
        pitch_area_min=1000,
        delivery_motion_min_px=4,
        delivery_motion_frames=2,
        pitch_stable_frames=2,
    )
    detector = EventDetector(cfg)
    pitch_cnt = np.array([[[400, 250]], [[700, 250]], [[700, 500]], [[400, 500]]], dtype=np.int32)

    state = DRSState()
    for y, y_prev in [(280, 270), (290, 280), (300, 290), (310, 300), (320, 310)]:
        state.ball.x = 500
        state.ball.y = y
        state.ball.x_prev = 500
        state.ball.y_prev = y_prev
        state.ball.prev_x_diff = 0
        state.ball.prev_y_diff = y - y_prev
        detector.process_frame(state, [pitch_cnt], [])

    assert state.pitch_point == (500, 310)


def test_impact_requires_pitch_point_first():
    import numpy as np

    cfg = DRSConfig(delivery_motion_min_px=4, delivery_motion_frames=1)
    detector = EventDetector(cfg)
    batsman_cnt = np.array([[[480, 350]], [[560, 350]], [[560, 520]], [[480, 520]]], dtype=np.int32)

    state = _moving_ball_state(400, 390)
    detector.process_frame(state, [], [batsman_cnt])

    assert state.impact_point is None


def test_impact_not_before_pitch_line():
    import numpy as np

    cfg = DRSConfig(delivery_motion_min_px=4, delivery_motion_frames=1, impact_batleg_px=25)
    detector = EventDetector(cfg)
    batsman_cnt = np.array([[[480, 350]], [[560, 350]], [[560, 520]], [[480, 520]]], dtype=np.int32)

    state = _moving_ball_state(300, 290)
    state.pitch_point = (500, 350)
    detector.process_frame(state, [], [batsman_cnt])

    assert state.impact_point is None


def test_delivery_latches_after_motion():
    cfg = DRSConfig(delivery_motion_min_px=4, delivery_motion_frames=2)
    detector = EventDetector(cfg)
    assert not detector._delivery_in_progress(_moving_ball_state(300, 295))
    assert detector._delivery_in_progress(_moving_ball_state(305, 300))
    assert detector._delivery_in_progress(DRSState())
