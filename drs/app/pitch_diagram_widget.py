"""Qt widget for top-down pitch diagram and side elevation animation."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from drs.app.image_utils import bgr_to_qimage
from drs.fusion.calibration import StumpPoints
from drs.ui.pitch_diagram import render_combined_diagram, render_pitch_diagram


class PitchDiagramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: list[tuple[float, float]] = []
        self._pixel_points: list[tuple[float, float]] = []
        self._live_ball: tuple[float, float] | None = None
        self._live_pixel: tuple[float, float] | None = None
        self._frame_h = 720
        self._stump_points: StumpPoints | None = None
        self._pitch_bounce: tuple[float, float] | None = None
        self._impact: tuple[float, float] | None = None
        self._progress = 0
        self._prev_point_count = 0
        self._label = QLabel("Pitch diagram")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(560, 360)
        self._scope_label = QLabel(
            "Side view: height hint only — corridor line assist is the primary verdict."
        )
        self._scope_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scope_label.setStyleSheet("color: #888; font-size: 11px;")
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._scope_label)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_trajectory(
        self,
        pitch_points: list[tuple[float, float]],
        *,
        pitch_bounce: tuple[float, float] | None = None,
        impact: tuple[float, float] | None = None,
        pixel_points: list[tuple[float, float]] | None = None,
        frame_h: int = 720,
        stump_points: StumpPoints | None = None,
        animate: bool = False,
        live_ball: tuple[float, float] | None = None,
        live_pixel: tuple[float, float] | None = None,
    ) -> None:
        self._points = list(pitch_points)
        self._pixel_points = list(pixel_points or [])
        self._live_ball = live_ball
        self._live_pixel = live_pixel
        self._frame_h = frame_h
        self._stump_points = stump_points
        self._pitch_bounce = pitch_bounce
        self._impact = impact
        self._timer.stop()
        if animate:
            self._progress = 0
            if len(self._points) > 1:
                self._timer.start(50)
        elif live_ball is not None:
            self._progress = max(len(self._points), 1) - 1
        elif len(self._points) > self._prev_point_count:
            self._progress = len(self._points) - 1
        self._prev_point_count = len(self._points)
        self._render()

    def _tick(self) -> None:
        if not self._points:
            return
        if self._progress < len(self._points) - 1:
            self._progress += 1
            self._render()
        else:
            self._timer.stop()

    def _render(self) -> None:
        progress = None if self._live_ball is not None else (self._progress if self._points else None)
        if self._pixel_points or self._live_ball is not None:
            img = render_combined_diagram(
                self._points,
                self._pixel_points,
                progress=progress,
                frame_h=self._frame_h,
                pitch_bounce=self._pitch_bounce,
                impact=self._impact,
                live_ball=self._live_ball,
                live_pixel=self._live_pixel,
                stump_points=self._stump_points,
            )
        else:
            img = render_pitch_diagram(
                self._points,
                progress=progress,
                pitch_bounce=self._pitch_bounce,
                impact=self._impact,
                live_ball=self._live_ball,
            )
        self._show_cv(img)

    def _show_cv(self, frame: np.ndarray) -> None:
        qimg = bgr_to_qimage(frame)
        self._label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self._label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        ))

    def clear(self) -> None:
        self._timer.stop()
        self._points.clear()
        self._pixel_points.clear()
        self._live_ball = None
        self._live_pixel = None
        self._label.setText("Pitch diagram")
        self._label.setPixmap(QPixmap())
