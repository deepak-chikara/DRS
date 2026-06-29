"""Factory for advisory service instances."""

from __future__ import annotations

from drs.config import DRSConfig
from drs.services.advisory.null_service import NullAdvisoryService
from drs.services.advisory.ollama_client import OllamaClient
from drs.services.advisory.ollama_service import OllamaAdvisoryService
from drs.services.advisory.protocol import IAdvisoryService


def create_advisory_service(config: DRSConfig) -> IAdvisoryService:
    if not config.ai_enabled:
        return NullAdvisoryService()

    provider = (config.ai_provider or "ollama").lower()
    if provider == "ollama":
        client = OllamaClient(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout_seconds=config.ollama_timeout_seconds,
            temperature=config.ollama_temperature,
        )
        return OllamaAdvisoryService(client, resolve_review=config.ai_resolve_review)

    return NullAdvisoryService()
