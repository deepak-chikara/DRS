"""Detection settings dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from drs.config import DRSConfig, save_config
from drs.paths import user_config_path


class SettingsDialog(QDialog):
    def __init__(self, config: DRSConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Detection Settings")
        self.setMinimumWidth(360)

        form = QFormLayout()
        self._scale = QDoubleSpinBox()
        self._scale.setRange(0.25, 1.0)
        self._scale.setSingleStep(0.05)
        self._scale.setValue(config.detection_scale)
        form.addRow("Detection scale", self._scale)

        self._pitch_min = QSpinBox()
        self._pitch_min.setRange(1000, 500000)
        self._pitch_min.setValue(config.pitch_area_min)
        form.addRow("Pitch area min", self._pitch_min)

        self._batsman_min = QSpinBox()
        self._batsman_min.setRange(500, 100000)
        self._batsman_min.setValue(config.batsman_area_min)
        form.addRow("Batsman area min", self._batsman_min)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Tune detection performance and sensitivity."))
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _save(self) -> None:
        self.config.detection_scale = self._scale.value()
        self.config.pitch_area_min = self._pitch_min.value()
        self.config.batsman_area_min = self._batsman_min.value()
        save_config(self.config, user_config_path())
        self.accept()
