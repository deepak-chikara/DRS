"""Deterministic confidence scoring for LBW decisions."""

from __future__ import annotations

from drs.services.advisory.models import ConfidenceReport


class ConfidenceScorer:
    """Score delivery evidence quality without LLM latency."""

    _SOURCE_WEIGHTS = {
        "yolo": 1.0,
        "color": 0.88,
        "tracker": 0.62,
        "none": 0.2,
        "fused": 0.9,
    }

    def score(
        self,
        *,
        ball_confidence: float,
        ball_source: str,
        trajectory_point_count: int,
        tracking_quality: str,
        stump_calibrated: bool,
        bat_leg_known: bool,
        pad_detected: bool = False,
        corridor_relative: float | None = None,
        ball_visible: bool = False,
        live_analysis: bool = False,
    ) -> ConfidenceReport:
        factors: list[str] = []

        src_key = ball_source.lower() if ball_source else "none"
        source_weight = self._SOURCE_WEIGHTS.get(src_key, 0.5)
        ball_detection = max(0.0, min(1.0, ball_confidence * source_weight))

        if src_key == "color" and ball_confidence >= 0.5:
            ball_detection = max(ball_detection, 0.72)
        if src_key == "yolo" and ball_confidence >= 0.25:
            ball_detection = max(ball_detection, ball_confidence)
        if src_key == "tracker" and ball_visible:
            ball_detection = max(ball_detection, 0.58)

        if ball_confidence < 0.35 and src_key not in ("color", "yolo"):
            factors.append("low ball detection confidence")
        if src_key == "tracker" and not ball_visible:
            factors.append("ball position from tracker gap-fill")

        if tracking_quality == "good":
            tracking = 1.0
        elif tracking_quality == "review":
            tracking = 0.68
        else:
            tracking = 0.45 if ball_visible else 0.30
            if not ball_visible:
                factors.append("insufficient ball tracking")

        if trajectory_point_count >= 5:
            tracking = max(tracking, 0.88)
        elif trajectory_point_count >= 2:
            tracking = max(tracking, 0.62)
        elif ball_visible and live_analysis:
            tracking = max(tracking, 0.55)

        calibration = 1.0 if stump_calibrated else 0.45
        if not stump_calibrated:
            factors.append("stump corridor not calibrated")

        if bat_leg_known:
            batsman = 1.0
        elif stump_calibrated and (pad_detected or ball_visible):
            batsman = 0.72
        else:
            batsman = 0.55
            if not bat_leg_known:
                factors.append("batsman contour not detected (non-blocking for corridor LBW)")

        overall = (
            ball_detection * 0.38
            + tracking * 0.32
            + calibration * 0.22
            + batsman * 0.08
        )

        if corridor_relative is not None and stump_calibrated:
            if 0.38 <= corridor_relative <= 0.62:
                overall += 0.08
            overall += 0.05

        if pad_detected and stump_calibrated:
            overall += 0.06

        if live_analysis and ball_visible and stump_calibrated:
            overall = max(overall, 0.58)

        overall = max(0.0, min(1.0, overall))

        return ConfidenceReport(
            overall=overall,
            ball_detection=ball_detection,
            tracking=tracking,
            calibration=calibration,
            batsman=batsman,
            factors=tuple(factors),
        )
