"""Detection and AI settings dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from drs.config import DRSConfig, save_config
from drs.paths import user_config_path
from drs.services.advisory.factory import create_advisory_service


class SettingsDialog(QDialog):
    def __init__(self, config: DRSConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Settings")
        self.setMinimumWidth(420)

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

        form.addRow(QLabel(""))
        form.addRow(QLabel("<b>AI Advisory (Ollama)</b>"))

        self._ai_enabled = QCheckBox("Enable AI Review panel and Ollama advisory")
        self._ai_enabled.setChecked(config.ai_enabled)
        form.addRow(self._ai_enabled)

        self._ai_live = QCheckBox("Analyze automatically during playback (live ball tracking)")
        self._ai_live.setChecked(config.ai_live_enabled)
        form.addRow(self._ai_live)

        self._ollama_model = QLineEdit(config.ollama_model)
        self._ollama_model.setPlaceholderText("llama3.2")
        form.addRow("Ollama model", self._ollama_model)

        self._ai_hint = QLabel(
            "Requires Ollama running locally: ollama pull llama3.2"
        )
        self._ai_hint.setWordWrap(True)
        self._ai_hint.setStyleSheet("color: #666;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        test_btn = buttons.addButton("Test Ollama", QDialogButtonBox.ButtonRole.ActionRole)
        test_btn.clicked.connect(self._test_ollama)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Tune detection performance and AI advisory."))
        layout.addLayout(form)
        layout.addWidget(self._ai_hint)
        layout.addWidget(buttons)

    def _test_ollama(self) -> None:
        self.config.ai_enabled = self._ai_enabled.isChecked()
        self.config.ollama_model = self._ollama_model.text().strip() or "llama3.2"
        service = create_advisory_service(self.config)
        if service.provider_name == "none":
            QMessageBox.warning(
                self,
                "Ollama test",
                "Enable AI advisory first (check the box above).",
            )
            return
        if service.is_available():
            QMessageBox.information(
                self,
                "Ollama test",
                f"Connected to Ollama.\nModel configured: {self.config.ollama_model}",
            )
        else:
            QMessageBox.warning(
                self,
                "Ollama test",
                "Cannot reach Ollama at http://127.0.0.1:11434\n\n"
                "1. Install from https://ollama.com\n"
                "2. Run: ollama pull llama3.2\n"
                "3. Ensure Ollama is running in the system tray",
            )

    def _save(self) -> None:
        self.config.detection_scale = self._scale.value()
        self.config.pitch_area_min = self._pitch_min.value()
        self.config.batsman_area_min = self._batsman_min.value()
        self.config.ai_enabled = self._ai_enabled.isChecked()
        self.config.ai_live_enabled = self._ai_live.isChecked()
        model = self._ollama_model.text().strip()
        if model:
            self.config.ollama_model = model
        save_config(self.config, user_config_path())
        self.accept()
