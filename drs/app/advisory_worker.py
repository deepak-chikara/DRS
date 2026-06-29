"""Background thread for AI advisory requests."""

from __future__ import annotations

import logging

from PySide6.QtCore import QMutex, QThread, Signal

from drs.services.advisory.models import AdvisoryResult, DeliveryEvidence
from drs.services.advisory.protocol import IAdvisoryService

logger = logging.getLogger("drs.app.advisory")


class AdvisoryWorker(QThread):
    advisory_ready = Signal(object)
    advisory_failed = Signal(str)
    provider_status = Signal(bool, str)

    def __init__(self, service: IAdvisoryService, parent=None):
        super().__init__(parent)
        self._service = service
        self._mutex = QMutex()
        self._pending: DeliveryEvidence | None = None
        self._last_key: tuple[int, str, int] | None = None
        self._force = False
        self._check_only = False

    @property
    def service(self) -> IAdvisoryService:
        return self._service

    def set_service(self, service: IAdvisoryService) -> None:
        self._service = service

    def check_provider(self) -> None:
        self._check_only = True
        if not self.isRunning():
            self.start()

    def analyze(self, evidence: DeliveryEvidence, *, force: bool = False) -> None:
        key = (evidence.delivery_id, evidence.cv_verdict, evidence.frame_pos)
        if not force and key == self._last_key:
            return
        self._mutex.lock()
        self._pending = evidence
        self._force = force
        self._mutex.unlock()
        if not self.isRunning():
            self.start()

    def run(self) -> None:
        if self._check_only:
            self._check_only = False
            available = self._service.is_available()
            label = "Connected" if available else "Offline — start Ollama (ollama serve)"
            self.provider_status.emit(available, label)
            return

        while True:
            self._mutex.lock()
            evidence = self._pending
            force = self._force
            self._pending = None
            self._force = False
            self._mutex.unlock()
            if evidence is None:
                break

            self._last_key = (evidence.delivery_id, evidence.cv_verdict, evidence.frame_pos)
            try:
                if self._service.provider_name == "none":
                    result = AdvisoryResult.fallback(
                        delivery_id=evidence.delivery_id,
                        cv_verdict=evidence.cv_verdict,
                        reason="AI advisory disabled — enable in Tools → Settings",
                    )
                else:
                    result = self._service.analyze(evidence)
                logger.info(
                    "Advisory delivery=%s cv=%s ai=%s confidence=%.2f latency=%.0fms",
                    evidence.delivery_id,
                    evidence.cv_verdict,
                    result.recommended_verdict,
                    result.confidence,
                    result.latency_ms,
                )
                self.advisory_ready.emit(result)
            except Exception as exc:
                logger.exception("Advisory worker error")
                self.advisory_failed.emit(str(exc))

            self._mutex.lock()
            has_more = self._pending is not None
            self._mutex.unlock()
            if not has_more:
                break
