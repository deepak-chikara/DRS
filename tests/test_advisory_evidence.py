"""Evidence builder tests."""

from drs.fusion.calibration import StumpPoints
from drs.services.advisory.evidence import build_delivery_evidence, infer_cv_verdict
from drs.state import DRSState


def _stumps():
    return StumpPoints((615, 221), (640, 221), (611, 495), (643, 492))


def test_infer_cv_verdict_on_wicket_line():
    state = DRSState()
    state.ball.x = 627
    state.ball.y = 376
    state.ball.confidence = 0.72
    state.ball.source = "color"
    verdict, reason = infer_cv_verdict(state, _stumps(), live=True)
    assert verdict == "REVIEW"
    assert "wicket line" in reason.lower() or "center" in reason.lower()


def test_build_live_evidence_without_formal_verdict():
    state = DRSState()
    state.ball.x = 627
    state.ball.y = 376
    state.ball.confidence = 0.72
    state.ball.source = "color"
    evidence = build_delivery_evidence(
        state,
        mode="file",
        frame_pos=50,
        delivery_id=1,
        stump_points=_stumps(),
        live=True,
    )
    assert evidence is not None
    assert evidence.live_analysis
    assert evidence.confidence_report.overall >= 0.55
