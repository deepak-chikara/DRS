"""Video display widget with verdict banner overlay."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from drs.app.image_utils import bgr_to_qimage

_VERDICT_COLORS = {
    "OUT": "#c62828",
    "NOT OUT": "#2e7d32",
    "REVIEW": "#f9a825",
}


class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel("Open a video to begin", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setMinimumSize(640, 360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self._verdict = ""
        self._confidence = 0
        self._ai_verdict = ""
        self._pixmap: QPixmap | None = None

    def show_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        img = bgr_to_qimage(frame)
        pix = QPixmap.fromImage(img)
        self._pixmap = pix
        self._refresh_display()

    def show_message(self, text: str) -> None:
        self._label.setPixmap(QPixmap())
        self._label.setText(text)
        self._pixmap = None

    def set_verdict_banner(
        self,
        verdict: str,
        *,
        confidence: int = 0,
        ai_verdict: str = "",
    ) -> None:
        self._verdict = verdict or ""
        self._confidence = confidence
        self._ai_verdict = ai_verdict or ""
        self._refresh_display()

    def clear_verdict_banner(self) -> None:
        self._verdict = ""
        self._confidence = 0
        self._ai_verdict = ""
        self._refresh_display()

    def _refresh_display(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if not self._verdict:
            self._label.setPixmap(scaled)
            return

        composed = QPixmap(scaled.size())
        composed.fill(Qt.GlobalColor.transparent)
        painter = QPainter(composed)
        painter.drawPixmap(0, 0, scaled)

        color = QColor(_VERDICT_COLORS.get(self._verdict, "#546e7a"))
        banner_h = max(48, composed.height() // 10)
        painter.fillRect(0, 0, composed.width(), banner_h, color)
        painter.setPen(Qt.GlobalColor.white)
        font = QFont()
        font.setBold(True)
        font.setPointSize(max(14, banner_h // 3))
        painter.setFont(font)
        text = self._verdict
        if self._confidence:
            text += f"  {self._confidence}%"
        if self._ai_verdict and self._ai_verdict != self._verdict:
            text += f"  · AI: {self._ai_verdict}"
        painter.drawText(12, banner_h - 12, text)
        painter.end()
        self._label.setPixmap(composed)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_display()
