"""Video display widget."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from drs.app.image_utils import bgr_to_qimage


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

    def show_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        img = bgr_to_qimage(frame)
        pix = QPixmap.fromImage(img)
        scaled = pix.scaled(self._label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._label.setPixmap(scaled)

    def show_message(self, text: str) -> None:
        self._label.setPixmap(QPixmap())
        self._label.setText(text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        pix = self._label.pixmap()
        if pix and not pix.isNull():
            self._label.setPixmap(pix.scaled(self._label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
