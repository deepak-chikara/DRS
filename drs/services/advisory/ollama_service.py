"""Ollama-backed AI advisory service."""

from __future__ import annotations

import json
import logging
import time

import httpx

from drs.services.advisory.models import AdvisoryResult, DeliveryEvidence, parse_advisory_json
from drs.services.advisory.ollama_client import OllamaClient
from drs.services.advisory.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger("drs.advisory.ollama")


class OllamaAdvisoryService:
    def __init__(self, client: OllamaClient, *, resolve_review: bool = True):
        self._client = client
        self._resolve_review = resolve_review

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._client.model

    def is_available(self) -> bool:
        return self._client.is_available()

    def analyze(self, evidence: DeliveryEvidence) -> AdvisoryResult:
        if not evidence.cv_verdict:
            return AdvisoryResult.fallback(
                delivery_id=evidence.delivery_id,
                cv_verdict="",
                reason="No CV verdict to analyze",
            )

        t0 = time.perf_counter()
        try:
            raw = self._client.chat_json(
                SYSTEM_PROMPT,
                build_user_prompt(json.dumps(evidence.to_dict(), indent=2)),
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            result = parse_advisory_json(
                raw,
                delivery_id=evidence.delivery_id,
                cv_verdict=evidence.cv_verdict,
                provider=self.provider_name,
                model=self._client.model,
                latency_ms=latency_ms,
            )
            return self._apply_policy(evidence, result)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            logger.warning("Ollama advisory failed: %s", exc)
            return AdvisoryResult.fallback(
                delivery_id=evidence.delivery_id,
                cv_verdict=evidence.cv_verdict,
                reason=f"AI unavailable — {exc}",
            )

    def _apply_policy(self, evidence: DeliveryEvidence, result: AdvisoryResult) -> AdvisoryResult:
        """AI may resolve REVIEW only; never override confident OUT/NOT OUT."""
        cv = evidence.cv_verdict
        ai = result.recommended_verdict

        if cv in ("OUT", "NOT OUT") and ai in ("OUT", "NOT OUT") and ai != cv:
            if evidence.confidence_report.overall >= 0.72:
                return AdvisoryResult(
                    recommended_verdict="REVIEW",
                    confidence=result.confidence,
                    summary=f"AI disagrees with CV ({cv}) — manual review recommended",
                    reasoning=result.reasoning + (f"CV says {cv}; AI suggests {ai}.",),
                    caveats=result.caveats,
                    provider=result.provider,
                    model=result.model,
                    latency_ms=result.latency_ms,
                    delivery_id=result.delivery_id,
                    cv_verdict=cv,
                    valid=result.valid,
                )

        if cv in ("OUT", "NOT OUT") and not self._resolve_review:
            if ai != cv and ai != "REVIEW":
                ai = "REVIEW"

        if cv == "REVIEW" and not self._resolve_review:
            return AdvisoryResult(
                recommended_verdict="REVIEW",
                confidence=result.confidence,
                summary=result.summary,
                reasoning=result.reasoning,
                caveats=result.caveats,
                provider=result.provider,
                model=result.model,
                latency_ms=result.latency_ms,
                delivery_id=result.delivery_id,
                cv_verdict=cv,
                valid=result.valid,
            )

        if (
            result.valid
            and evidence.corridor_relative is not None
            and 0.38 <= evidence.corridor_relative <= 0.62
            and evidence.stump_calibrated
            and result.recommended_verdict == "REVIEW"
            and result.confidence < 0.55
            and evidence.confidence_report.overall >= 0.55
        ):
            return AdvisoryResult(
                recommended_verdict=result.recommended_verdict,
                confidence=max(result.confidence, 0.62),
                summary=result.summary,
                reasoning=result.reasoning,
                caveats=result.caveats,
                provider=result.provider,
                model=result.model,
                latency_ms=result.latency_ms,
                delivery_id=result.delivery_id,
                cv_verdict=cv,
                valid=result.valid,
            )

        return result
