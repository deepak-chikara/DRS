"""AI advisory services for LBW decision support."""

from drs.services.advisory.models import AdvisoryResult, ConfidenceReport, DeliveryEvidence

__all__ = [
    "AdvisoryResult",
    "ConfidenceReport",
    "DeliveryEvidence",
    "create_advisory_service",
]


def create_advisory_service(config):
    from drs.services.advisory.factory import create_advisory_service as _create
    return _create(config)
