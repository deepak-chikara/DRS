"""AI Review dock panel for DRS Pro."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from drs.services.advisory.models import AdvisoryResult, ConfidenceReport, DeliveryEvidence


class AdvisoryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._status = QLabel("AI: Disabled")
        self._status.setStyleSheet("font-weight: bold;")
        self._cv_verdict = QLabel("CV verdict: —")
        self._confidence_bar = QProgressBar()
        self._confidence_bar.setRange(0, 100)
        self._confidence_bar.setValue(0)
        self._confidence_label = QLabel("Confidence: —")
        self._ai_verdict = QLabel("AI recommendation: —")
        self._ai_verdict.setWordWrap(True)
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._reasoning = QLabel("")
        self._reasoning.setWordWrap(True)
        self._caveats = QLabel("")
        self._caveats.setWordWrap(True)
        self._caveats.setStyleSheet("color: #888;")
        self._clip = QLabel("")
        self._clip.setWordWrap(True)
        self._disagreement = QLabel("")
        self._disagreement.setWordWrap(True)
        self._last_cv_verdict = ""

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._status)
        layout.addWidget(self._sep())
        layout.addWidget(self._cv_verdict)
        layout.addWidget(self._disagreement)
        layout.addWidget(self._confidence_label)
        layout.addWidget(self._confidence_bar)
        layout.addWidget(self._sep())
        layout.addWidget(QLabel("AI Recommendation"))
        layout.addWidget(self._ai_verdict)
        layout.addWidget(self._summary)
        layout.addWidget(QLabel("Reasoning"))
        layout.addWidget(self._reasoning)
        layout.addWidget(QLabel("Caveats"))
        layout.addWidget(self._caveats)
        layout.addWidget(self._clip)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.addWidget(scroll)

    @staticmethod
    def _sep() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    def set_enabled(self, enabled: bool) -> None:
        if not enabled:
            self._status.setText("AI: Disabled")
            self._status.setStyleSheet("font-weight: bold; color: #888;")

    def set_provider_status(self, available: bool, label: str) -> None:
        color = "#2ecc71" if available else "#e74c3c"
        self._status.setText(f"Ollama: {label}")
        self._status.setStyleSheet(f"font-weight: bold; color: {color};")

    def set_analyzing(self) -> None:
        self._status.setText("Ollama: Analyzing…")
        self._status.setStyleSheet("font-weight: bold; color: #f39c12;")
        self._ai_verdict.setText("AI recommendation: Analyzing…")
        self._ai_verdict.setStyleSheet("color: #f39c12; font-weight: bold;")

    def show_evidence(self, evidence: DeliveryEvidence) -> None:
        self._last_cv_verdict = evidence.cv_verdict
        live_tag = " [live]" if getattr(evidence, "live_analysis", False) else ""
        self._cv_verdict.setText(
            f"CV verdict: {evidence.cv_verdict}{live_tag} — {evidence.cv_reason}"
        )
        self._disagreement.setText("")
        self._disagreement.setStyleSheet("")
        self._apply_confidence(evidence.confidence_report)
        if getattr(evidence, "corridor_assessment", None) and evidence.corridor_assessment != "unknown":
            self._summary.setText(f"Line: {evidence.corridor_assessment}")
        if evidence.clip_path:
            self._clip.setText(f"Clip: {evidence.clip_path}")
        else:
            self._clip.setText("")

    def show_result(self, result: AdvisoryResult) -> None:
        verdict_style = {
            "OUT": "color: #e74c3c; font-weight: bold;",
            "NOT OUT": "color: #2ecc71; font-weight: bold;",
            "REVIEW": "color: #f39c12; font-weight: bold;",
        }
        style = verdict_style.get(result.recommended_verdict, "")
        self._ai_verdict.setText(f"AI recommendation: {result.recommended_verdict}")
        self._ai_verdict.setStyleSheet(style)
        cv = self._last_cv_verdict or "—"
        if cv in ("OUT", "NOT OUT", "REVIEW") and result.recommended_verdict != cv:
            self._disagreement.setText(
                f"⚠ CV vs AI disagree: CV={cv}, AI={result.recommended_verdict}"
            )
            self._disagreement.setStyleSheet(
                "color: #c62828; font-weight: bold; background: #fff3e0; padding: 4px;"
            )
        else:
            self._disagreement.setText("CV and AI agree" if cv == result.recommended_verdict else "")
            self._disagreement.setStyleSheet("color: #2e7d32; font-weight: bold;")
        conf_pct = int(result.confidence * 100)
        self._summary.setText(f"{result.summary} ({conf_pct}% AI confidence)")
        bullets = "\n".join(f"• {r}" for r in result.reasoning)
        self._reasoning.setText(bullets or "—")
        caveats = "\n".join(f"• {c}" for c in result.caveats)
        self._caveats.setText(caveats or "—")
        if result.valid:
            self.set_provider_status(True, "Connected")

    def show_error(self, message: str) -> None:
        self._ai_verdict.setText("AI recommendation: REVIEW")
        self._ai_verdict.setStyleSheet("color: #f39c12; font-weight: bold;")
        self._summary.setText(message)
        self.set_provider_status(False, "Offline")

    def clear(self) -> None:
        self._cv_verdict.setText("CV verdict: —")
        self._confidence_bar.setValue(0)
        self._confidence_label.setText("Confidence: —")
        self._ai_verdict.setText("AI recommendation: —")
        self._ai_verdict.setStyleSheet("")
        self._summary.setText("")
        self._reasoning.setText("")
        self._caveats.setText("")
        self._clip.setText("")

    def _apply_confidence(self, report: ConfidenceReport) -> None:
        pct = int(report.overall * 100)
        self._confidence_bar.setValue(pct)
        self._confidence_label.setText(f"Evidence confidence: {pct}%")
        if report.overall < 0.72:
            self._confidence_bar.setStyleSheet("QProgressBar::chunk { background: #f39c12; }")
        else:
            self._confidence_bar.setStyleSheet("QProgressBar::chunk { background: #2ecc71; }")
