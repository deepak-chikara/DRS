"""Verdict engine tests."""

from drs.config import DRSConfig
from drs.decision.verdict import VerdictEngine
from drs.fusion.calibration import StumpPoints
from drs.state import DRSState


def _stumps():
    return StumpPoints((100, 500), (200, 500), (110, 100), (190, 100))


def _confident_state(state: DRSState) -> None:
    state.ball.confidence = 0.9
    state.ball.source = "yolo"
    state.trajectory_pitch_points = [
        (0.5, 0.9), (0.5, 0.7), (0.5, 0.5), (0.5, 0.3), (0.5, 0.1),
    ]


def test_verdict_out_inside_corridor():
    state = DRSState()
    state.pad_detected = True
    state.ball.x = 150
    state.ball.y = 300
    state.bat_leg = 400
    _confident_state(state)
    engine = VerdictEngine(DRSConfig())
    engine.evaluate(state, _stumps())
    assert state.verdict == "OUT"


def test_verdict_not_out_outside_corridor():
    state = DRSState()
    state.pad_detected = True
    state.ball.x = 50
    state.ball.y = 300
    state.bat_leg = 400
    _confident_state(state)
    engine = VerdictEngine(DRSConfig())
    engine.evaluate(state, _stumps())
    assert state.verdict == "NOT OUT"


def test_verdict_review_no_ball():
    state = DRSState()
    state.pad_detected = True
    state.bat_leg = 400
    engine = VerdictEngine(DRSConfig())
    engine.evaluate(state, _stumps())
    assert state.verdict == "REVIEW"


def test_verdict_downgrade_low_confidence():
    state = DRSState()
    state.pad_detected = True
    state.ball.x = 150
    state.ball.y = 300
    state.bat_leg = 400
    state.ball.confidence = 0.1
    state.ball.source = "tracker"
    engine = VerdictEngine(DRSConfig())
    engine.evaluate(state, _stumps())
    assert state.verdict == "REVIEW"
    assert "confidence" in state.verdict_reason.lower()
