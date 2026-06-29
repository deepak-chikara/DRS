"""No-op advisory service when AI is disabled."""

from __future__ import annotations

from drs.services.advisory.models import AdvisoryResult, DeliveryEvidence


class NullAdvisoryService:
    @property
    def provider_name(self) -> str:
        return "none"

    def is_available(self) -> bool:
        return False

    def analyze(self, evidence: DeliveryEvidence) -> AdvisoryResult:
        return AdvisoryResult.fallback(
            delivery_id=evidence.delivery_id,
            cv_verdict=evidence.cv_verdict,
            reason="AI advisory disabled",
        )
