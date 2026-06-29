"""Advisory service interface."""

from __future__ import annotations

from typing import Protocol

from drs.services.advisory.models import AdvisoryResult, DeliveryEvidence


class IAdvisoryService(Protocol):
    """Pluggable AI advisory provider."""

    @property
    def provider_name(self) -> str:
        ...

    def is_available(self) -> bool:
        """True when the provider can accept requests."""
        ...

    def analyze(self, evidence: DeliveryEvidence) -> AdvisoryResult:
        """Synchronous analysis — call from background thread only."""
        ...
