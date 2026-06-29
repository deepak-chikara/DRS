"""Ollama advisory service tests with mocked HTTP."""

import json
from dataclasses import replace

import httpx

from drs.services.advisory.confidence import ConfidenceReport
from drs.services.advisory.models import DeliveryEvidence
from drs.services.advisory.ollama_client import OllamaClient
from drs.services.advisory.ollama_service import OllamaAdvisoryService


def _sample_evidence(cv_verdict: str = "REVIEW") -> DeliveryEvidence:
    report = ConfidenceReport(
        overall=0.55,
        ball_detection=0.6,
        tracking=0.5,
        calibration=1.0,
        batsman=1.0,
    )
    return DeliveryEvidence(
        mode="file",
        frame_pos=99,
        delivery_id=1,
        cv_verdict=cv_verdict,
        cv_reason="test",
        ball_x=627,
        ball_y=376,
        ball_source="color",
        ball_confidence=0.6,
        impact_point=(627, 400),
        pitch_point=(627, 500),
        motion_class="Pad",
        pad_detected=True,
        trajectory_point_count=4,
        tracking_quality="review",
        stump_calibrated=True,
        corridor_relative=0.5,
        confidence_report=report,
    )


def test_ollama_client_is_available_mock():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3.2"}]})

    transport = httpx.MockTransport(handler)
    client = OllamaClient()
    with httpx.Client(transport=transport, base_url=client.base_url) as http:
        resp = http.get(f"{client.base_url}/api/tags")
        assert resp.status_code == 200


def test_ollama_service_analyze_mock():
    ai_json = json.dumps({
        "recommended_verdict": "OUT",
        "confidence": 0.8,
        "summary": "Pad on line.",
        "reasoning": ["Good evidence"],
        "caveats": ["Height not checked"],
    })

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": ai_json}})
        return httpx.Response(404)

    service = OllamaAdvisoryService(
        OllamaClient(base_url="http://test", model="llama3.2", timeout_seconds=5),
        resolve_review=True,
    )
    service._client.chat_json = lambda _s, _u: ai_json  # type: ignore[method-assign]

    result = service.analyze(_sample_evidence("REVIEW"))
    assert result.recommended_verdict == "OUT"
    assert result.valid


def test_ollama_service_wont_override_confident_cv_disagreement():
    ai_json = json.dumps({
        "recommended_verdict": "NOT OUT",
        "confidence": 0.9,
        "summary": "Outside line.",
        "reasoning": ["Disagrees"],
        "caveats": [],
    })
    service = OllamaAdvisoryService(
        OllamaClient(base_url="http://test", model="llama3.2"),
        resolve_review=True,
    )
    service._client.chat_json = lambda _s, _u: ai_json  # type: ignore[method-assign]

    evidence = _sample_evidence("OUT")
    evidence = replace(
        evidence,
        confidence_report=ConfidenceReport(
            overall=0.85, ball_detection=0.9, tracking=0.9, calibration=1.0, batsman=1.0,
        ),
    )
    result = service.analyze(evidence)
    assert result.recommended_verdict == "REVIEW"
