"""Confidence scorer tests."""

from drs.services.advisory.confidence import ConfidenceScorer


def test_confidence_high_with_yolo_and_good_tracking():
    scorer = ConfidenceScorer()
    report = scorer.score(
        ball_confidence=0.9,
        ball_source="yolo",
        trajectory_point_count=8,
        tracking_quality="good",
        stump_calibrated=True,
        bat_leg_known=True,
    )
    assert report.overall >= 0.85


def test_confidence_low_with_tracker_and_sparse_track():
    scorer = ConfidenceScorer()
    report = scorer.score(
        ball_confidence=0.3,
        ball_source="tracker",
        trajectory_point_count=1,
        tracking_quality="insufficient",
        stump_calibrated=False,
        bat_leg_known=False,
        ball_visible=False,
    )
    assert report.overall < 0.5
    assert len(report.factors) >= 2


def test_confidence_live_ball_on_wicket_line():
    scorer = ConfidenceScorer()
    report = scorer.score(
        ball_confidence=0.72,
        ball_source="color",
        trajectory_point_count=1,
        tracking_quality="insufficient",
        stump_calibrated=True,
        bat_leg_known=False,
        corridor_relative=0.5,
        ball_visible=True,
        live_analysis=True,
    )
    assert report.overall >= 0.58
