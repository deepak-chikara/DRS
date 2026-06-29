"""Immutable DTOs for AI advisory pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


VALID_VERDICTS = frozenset({"OUT", "NOT OUT", "REVIEW"})


@dataclass(frozen=True)
class ConfidenceReport:
    overall: float
    ball_detection: float
    tracking: float
    calibration: float
    batsman: float
    factors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "ball_detection": self.ball_detection,
            "tracking": self.tracking,
            "calibration": self.calibration,
            "batsman": self.batsman,
            "factors": list(self.factors),
        }


@dataclass(frozen=True)
class DeliveryEvidence:
    mode: str
    frame_pos: int
    delivery_id: int
    cv_verdict: str
    cv_reason: str
    ball_x: int
    ball_y: int
    ball_source: str
    ball_confidence: float
    impact_point: tuple[int, int] | None
    pitch_point: tuple[int, int] | None
    motion_class: str
    pad_detected: bool
    trajectory_point_count: int
    tracking_quality: str
    stump_calibrated: bool
    corridor_relative: float | None
    confidence_report: ConfidenceReport
    clip_path: str | None = None
    corridor_assessment: str = "unknown"
    live_analysis: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "frame_pos": self.frame_pos,
            "delivery_id": self.delivery_id,
            "cv_verdict": self.cv_verdict,
            "cv_reason": self.cv_reason,
            "ball": {
                "x": self.ball_x,
                "y": self.ball_y,
                "source": self.ball_source,
                "confidence": self.ball_confidence,
            },
            "impact_point": list(self.impact_point) if self.impact_point else None,
            "pitch_point": list(self.pitch_point) if self.pitch_point else None,
            "motion_class": self.motion_class,
            "pad_detected": self.pad_detected,
            "trajectory_point_count": self.trajectory_point_count,
            "tracking_quality": self.tracking_quality,
            "stump_calibrated": self.stump_calibrated,
            "corridor_relative": self.corridor_relative,
            "corridor_assessment": self.corridor_assessment,
            "live_analysis": self.live_analysis,
            "confidence_report": self.confidence_report.to_dict(),
            "clip_path": self.clip_path,
        }


@dataclass(frozen=True)
class AdvisoryResult:
    recommended_verdict: str
    confidence: float
    summary: str
    reasoning: tuple[str, ...]
    caveats: tuple[str, ...]
    provider: str
    model: str
    latency_ms: float
    delivery_id: int = 0
    cv_verdict: str = ""
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_verdict": self.recommended_verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "reasoning": list(self.reasoning),
            "caveats": list(self.caveats),
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "delivery_id": self.delivery_id,
            "cv_verdict": self.cv_verdict,
            "valid": self.valid,
        }

    @classmethod
    def fallback(cls, *, delivery_id: int, cv_verdict: str, reason: str) -> AdvisoryResult:
        return cls(
            recommended_verdict="REVIEW",
            confidence=0.0,
            summary=reason,
            reasoning=(reason,),
            caveats=("Automated AI review unavailable.",),
            provider="none",
            model="",
            latency_ms=0.0,
            delivery_id=delivery_id,
            cv_verdict=cv_verdict,
            valid=False,
        )


def parse_advisory_json(
    raw: str,
    *,
    delivery_id: int,
    cv_verdict: str,
    provider: str,
    model: str,
    latency_ms: float,
) -> AdvisoryResult:
    """Parse and validate LLM JSON response."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return AdvisoryResult.fallback(
            delivery_id=delivery_id,
            cv_verdict=cv_verdict,
            reason="AI response invalid — manual review recommended",
        )

    verdict = str(data.get("recommended_verdict", "REVIEW")).upper()
    if verdict not in VALID_VERDICTS:
        verdict = "REVIEW"

    confidence = float(data.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    summary = str(data.get("summary", "")).strip() or "No summary provided."
    reasoning_raw = data.get("reasoning") or []
    if isinstance(reasoning_raw, str):
        reasoning_raw = [reasoning_raw]
    reasoning = tuple(str(r).strip() for r in reasoning_raw[:5] if str(r).strip())

    caveats_raw = data.get("caveats") or []
    if isinstance(caveats_raw, str):
        caveats_raw = [caveats_raw]
    caveats = tuple(str(c).strip() for c in caveats_raw[:5] if str(c).strip())
    if not caveats:
        caveats = (
            "Height above stumps not assessed.",
            "Bat before pad not assessed.",
            "Pitching outside off not assessed.",
        )

    return AdvisoryResult(
        recommended_verdict=verdict,
        confidence=confidence,
        summary=summary,
        reasoning=reasoning or (summary,),
        caveats=caveats,
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        delivery_id=delivery_id,
        cv_verdict=cv_verdict,
        valid=True,
    )
