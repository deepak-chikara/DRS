"""OUT / NOT OUT / REVIEW verdict for LBW pad decisions."""

from __future__ import annotations

from drs.config import DRSConfig
from drs.fusion.calibration import StumpPoints, is_inside_corridor
from drs.state import DRSState


class VerdictEngine:
    """Evaluate whether a pad impact is inside the stump corridor."""

    def __init__(self, config: DRSConfig):
        self.config = config

    def evaluate(
        self,
        state: DRSState,
        stump_points: StumpPoints | None,
    ) -> None:
        pad_event = state.pad_detected or (
            state.impact_locked and state.last_motion_class == "Pad"
        )
        if not pad_event:
            return

        x = state.ball.x
        y = state.ball.y
        if state.impact_point is not None:
            x, y = state.impact_point

        if x == 0 and y == 0:
            state.verdict = "REVIEW"
            state.verdict_reason = "Pad suspected but ball position unknown"
            return

        if state.bat_leg == 10000:
            state.verdict = "REVIEW"
            state.verdict_reason = "Pad suspected but batsman not detected"
            return

        if stump_points is None or not stump_points.is_valid():
            state.verdict = "REVIEW"
            state.verdict_reason = "Pad suspected but stump corridor not set (calibrate stumps)"
            return

        inside = is_inside_corridor(x, y, stump_points)
        if inside:
            state.verdict = "OUT"
            state.verdict_reason = "Pad contact inside stump corridor"
        else:
            state.verdict = "NOT OUT"
            state.verdict_reason = "Pad contact outside stump corridor"
