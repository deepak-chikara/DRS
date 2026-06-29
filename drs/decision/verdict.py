"""OUT / NOT OUT / REVIEW verdict for LBW pad decisions."""

from __future__ import annotations

from drs.config import DRSConfig
from drs.fusion.calibration import StumpPoints, is_inside_corridor
from drs.services.advisory.confidence import ConfidenceScorer
from drs.state import DRSState
from drs.ui.pitch_diagram import tracking_quality


class VerdictEngine:
    """Evaluate whether a pad impact is inside the stump corridor."""

    def __init__(self, config: DRSConfig):
        self.config = config
        self._scorer = ConfidenceScorer()

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
            self._apply_confidence(state, stump_points)
            return

        if state.bat_leg == 10000:
            state.verdict = "REVIEW"
            state.verdict_reason = "Pad suspected but batsman not detected"
            self._apply_confidence(state, stump_points)
            return

        if stump_points is None or not stump_points.is_valid():
            state.verdict = "REVIEW"
            state.verdict_reason = "Pad suspected but stump corridor not set (calibrate stumps)"
            self._apply_confidence(state, stump_points)
            return

        inside = is_inside_corridor(x, y, stump_points)
        if inside:
            state.verdict = "OUT"
            state.verdict_reason = "Pad contact inside stump corridor"
        else:
            state.verdict = "NOT OUT"
            state.verdict_reason = "Pad contact outside stump corridor"

        self._apply_confidence(state, stump_points)

    def _apply_confidence(
        self,
        state: DRSState,
        stump_points: StumpPoints | None,
    ) -> None:
        traj_count = len(state.trajectory_pitch_points)
        report = self._scorer.score(
            ball_confidence=state.ball.confidence,
            ball_source=state.ball.source,
            trajectory_point_count=traj_count,
            tracking_quality=tracking_quality(traj_count),
            stump_calibrated=stump_points is not None and stump_points.is_valid(),
            bat_leg_known=state.bat_leg != 10000,
        )
        state.confidence_overall = report.overall

        min_conf = self.config.ai_min_confidence_auto
        if (
            state.verdict in ("OUT", "NOT OUT")
            and report.overall < min_conf
        ):
            state.verdict = "REVIEW"
            state.verdict_reason = (
                f"Low tracking confidence ({int(report.overall * 100)}%) — manual review recommended"
            )
