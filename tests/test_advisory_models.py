"""Advisory model validation tests."""

import json

from drs.services.advisory.models import AdvisoryResult, parse_advisory_json


def test_parse_advisory_json_valid():
    raw = json.dumps({
        "recommended_verdict": "OUT",
        "confidence": 0.88,
        "summary": "Pad contact on wicket line with good tracking.",
        "reasoning": ["Ball tracked across delivery", "Impact inside corridor"],
        "caveats": ["Height not checked"],
    })
    result = parse_advisory_json(
        raw,
        delivery_id=1,
        cv_verdict="REVIEW",
        provider="ollama",
        model="llama3.2",
        latency_ms=1200.0,
    )
    assert result.valid
    assert result.recommended_verdict == "OUT"
    assert result.confidence == 0.88
    assert len(result.reasoning) == 2


def test_parse_advisory_json_invalid_fallback():
    result = parse_advisory_json(
        "not json",
        delivery_id=2,
        cv_verdict="OUT",
        provider="ollama",
        model="test",
        latency_ms=0,
    )
    assert not result.valid
    assert result.recommended_verdict == "REVIEW"


def test_advisory_result_fallback():
    fb = AdvisoryResult.fallback(delivery_id=3, cv_verdict="OUT", reason="offline")
    assert fb.recommended_verdict == "REVIEW"
    assert not fb.valid
