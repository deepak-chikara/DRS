"""Timeline and transport controls."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)


class TimelineWidget(QWidget):
    play_clicked = Signal()
    pause_clicked = Signal()
    step_back_clicked = Signal()
    step_forward_clicked = Signal()
    restart_clicked = Signal()
    position_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.sliderReleased.connect(self._on_slider_released)

        self._frame_label = QLabel("Frame 0/0")
        self._btn_play = QPushButton("Play")
        self._btn_pause = QPushButton("Pause")
        self._btn_back = QPushButton("<<")
        self._btn_fwd = QPushButton(">>")
        self._btn_restart = QPushButton("Restart")

        self._btn_play.clicked.connect(self.play_clicked.emit)
        self._btn_pause.clicked.connect(self.pause_clicked.emit)
        self._btn_back.clicked.connect(self.step_back_clicked.emit)
        self._btn_fwd.clicked.connect(self.step_forward_clicked.emit)
        self._btn_restart.clicked.connect(self.restart_clicked.emit)

        row = QHBoxLayout(self)
        row.addWidget(self._btn_play)
        row.addWidget(self._btn_pause)
        row.addWidget(self._btn_back)
        row.addWidget(self._btn_fwd)
        row.addWidget(self._btn_restart)
        row.addWidget(self._slider, stretch=1)
        row.addWidget(self._frame_label)

        self._updating = False

    def _on_slider_released(self) -> None:
        self.position_changed.emit(self._slider.value())

    def set_total(self, total: int) -> None:
        self._updating = True
        self._slider.setMaximum(max(0, total - 1))
        self._updating = False

    def set_position(self, pos: int, total: int) -> None:
        self._updating = True
        self._slider.setValue(pos)
        self._frame_label.setText(f"Frame {pos + 1}/{max(total, 1)}")
        self._updating = False
