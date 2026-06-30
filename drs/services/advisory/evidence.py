"""Build immutable delivery evidence snapshots from engine state."""

from __future__ import annotations

from drs.fusion.calibration import StumpPoints, corridor_bounds_at_y, is_inside_corridor
from drs.services.advisory.confidence import ConfidenceScorer
from drs.services.advisory.models import DeliveryEvidence
from drs.state import DRSState
from drs.ui.pitch_diagram import tracking_quality


def corridor_relative(x: int, y: int, stumps: StumpPoints | None) -> float | None:
    if stumps is None or not stumps.is_valid():
        return None
    x_min, x_max = corridor_bounds_at_y(stumps, float(y))
    width = max(x_max - x_min, 1.0)
    return max(0.0, min(1.0, (float(x) - x_min) / width))


def corridor_assessment(rel: float | None) -> str:
    if rel is None:
        return "unknown"
    if 0.42 <= rel <= 0.58:
        return "on wicket line (center of stump corridor)"
    if rel < 0.42:
        return "off-side of stump corridor"
    return "leg-side of stump corridor"


def infer_cv_verdict(
    state: DRSState,
    stump_points: StumpPoints | None,
    *,
    live: bool = False,
) -> tuple[str, str]:
    """Provisional OUT/NOT OUT/REVIEW from geometry when full verdict not latched yet."""
    x = state.ball.x
    y = state.ball.y
    if state.impact_point is not None:
        x, y = state.impact_point

    stump_ok = stump_points is not None and stump_points.is_valid()
    pad_event = state.pad_detected or state.last_motion_class == "Pad"

    if pad_event and stump_ok and x != 0 and y != 0:
        inside = is_inside_corridor(x, y, stump_points)
        if inside:
            return "OUT", "Provisional: pad contact inside stump corridor"
        return "NOT OUT", "Provisional: pad contact outside stump corridor"

    rel = corridor_relative(x, y, stump_points) if x and y else None
    if stump_ok and rel is not None and state.ball.x != 0:
        line = corridor_assessment(rel)
        prefix = "Live tracking" if live else "Analysis"
        if 0.38 <= rel <= 0.62:
            return "REVIEW", f"{prefix}: ball {line} — likely hitting wickets if pad contact"
        return "REVIEW", f"{prefix}: ball {line}"

    if live and state.ball.x != 0:
        return "REVIEW", "Live ball tracking — awaiting pad/impact moment"

    if live:
        traj = len(state.trajectory_pitch_points)
        if stump_ok and traj == 0 and state.ball.x == 0:
            return "REVIEW", "Live analysis — ball not detected yet; stumps calibrated"
        return "REVIEW", "Live analysis — waiting for ball detection"

    return "REVIEW", "Manual AI analysis — play to pad/LBW moment or scrub to impact frame"


def build_delivery_evidence(
    state: DRSState,
    *,
    mode: str,
    frame_pos: int,
    delivery_id: int,
    stump_points: StumpPoints | None,
    clip_path: str | None = None,
    scorer: ConfidenceScorer | None = None,
    manual: bool = False,
    live: bool = False,
) -> DeliveryEvidence | None:
    ball_visible = state.ball.x != 0 or state.ball.y != 0
    if not state.verdict and not manual and not live:
        return None
    if live and not ball_visible and not state.verdict and len(state.trajectory_pitch_points) == 0:
        if stump_points is None or not stump_points.is_valid():
            return None

    if state.verdict:
        cv_verdict = state.verdict
        cv_reason = state.verdict_reason
    else:
        cv_verdict, cv_reason = infer_cv_verdict(state, stump_points, live=live or manual)

    traj_count = len(state.trajectory_pitch_points)
    if traj_count == 0 and ball_visible:
        traj_count = 1
    tq = tracking_quality(len(state.trajectory_pitch_points) or (1 if ball_visible else 0))
    stump_ok = stump_points is not None and stump_points.is_valid()
    bat_known = state.bat_leg != 10000

    scorer = scorer or ConfidenceScorer()

    x = state.ball.x
    y = state.ball.y
    if state.impact_point is not None:
        x, y = state.impact_point

    rel = corridor_relative(x, y, stump_points)

    confidence_report = scorer.score(
        ball_confidence=state.ball.confidence,
        ball_source=state.ball.source,
        trajectory_point_count=traj_count,
        tracking_quality=tq,
        stump_calibrated=stump_ok,
        bat_leg_known=bat_known,
        pad_detected=state.pad_detected,
        corridor_relative=rel,
        ball_visible=ball_visible,
        live_analysis=live,
    )

    return DeliveryEvidence(
        mode=mode,
        frame_pos=frame_pos,
        delivery_id=delivery_id,
        cv_verdict=cv_verdict,
        cv_reason=cv_reason,
        ball_x=state.ball.x,
        ball_y=state.ball.y,
        ball_source=state.ball.source,
        ball_confidence=state.ball.confidence,
        impact_point=state.impact_point,
        pitch_point=state.pitch_point,
        motion_class=state.last_motion_class,
        pad_detected=state.pad_detected,
        trajectory_point_count=traj_count,
        tracking_quality=tq,
        stump_calibrated=stump_ok,
        corridor_relative=rel,
        corridor_assessment=corridor_assessment(rel),
        confidence_report=confidence_report,
        clip_path=clip_path,
        live_analysis=live,
    )
