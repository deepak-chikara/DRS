"""Detection and AI settings dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from drs.config import save_config
from drs.config import DRSConfig
from drs.grounds import list_ground_ids, save_ground_preset, switch_ground
from drs.paths import user_config_path
from drs.services.advisory.factory import create_advisory_service


class SettingsDialog(QDialog):
    def __init__(self, config: DRSConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)

        form = QFormLayout()

        self._ground = QComboBox()
        self._ground.setEditable(True)
        for gid in list_ground_ids():
            self._ground.addItem(gid)
        idx = self._ground.findText(config.ground_id)
        if idx >= 0:
            self._ground.setCurrentIndex(idx)
        else:
            self._ground.setEditText(config.ground_id)
        form.addRow("Ground", self._ground)

        self._detection_mode = QComboBox()
        self._detection_mode.addItems(["color", "hybrid", "yolo"])
        dm_idx = self._detection_mode.findText(config.detection_mode)
        if dm_idx >= 0:
            self._detection_mode.setCurrentIndex(dm_idx)
        form.addRow("Detection mode", self._detection_mode)

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

        hsv = config.ball_hsv or {
            "hmin": 10, "smin": 44, "vmin": 192,
            "hmax": 125, "smax": 114, "vmax": 255,
        }
        self._hsv_fields: dict[str, QSpinBox] = {}
        hsv_box = QGroupBox("Ball color (HSV)")
        hsv_form = QFormLayout(hsv_box)
        for key, default in (
            ("hmin", 10), ("smin", 44), ("vmin", 192),
            ("hmax", 125), ("smax", 114), ("vmax", 255),
        ):
            spin = QSpinBox()
            spin.setRange(0, 255)
            spin.setValue(int(hsv.get(key, default)))
            self._hsv_fields[key] = spin
            hsv_form.addRow(key, spin)

        clip_box = QGroupBox("Live clips")
        clip_form = QFormLayout(clip_box)
        self._pre_roll = QDoubleSpinBox()
        self._pre_roll.setRange(2, 30)
        self._pre_roll.setValue(config.clip_pre_roll_seconds)
        clip_form.addRow("Pre-roll (s)", self._pre_roll)
        self._post_roll = QDoubleSpinBox()
        self._post_roll.setRange(2, 30)
        self._post_roll.setValue(config.clip_post_roll_seconds)
        clip_form.addRow("Post-roll (s)", self._post_roll)
        self._buffer = QSpinBox()
        self._buffer.setRange(15, 120)
        self._buffer.setValue(config.ring_buffer_seconds)
        clip_form.addRow("Ring buffer (s)", self._buffer)

        ai_box = QGroupBox("AI Advisory (Ollama)")
        ai_form = QFormLayout(ai_box)
        self._ai_enabled = QCheckBox("Enable AI Review panel and Ollama advisory")
        self._ai_enabled.setChecked(config.ai_enabled)
        ai_form.addRow(self._ai_enabled)
        self._ai_live = QCheckBox("Analyze automatically during playback")
        self._ai_live.setChecked(config.ai_live_enabled)
        ai_form.addRow(self._ai_live)
        self._ai_skip = QCheckBox("Skip AI when CV verdict is confident (OUT/NOT OUT ≥ 80%)")
        self._ai_skip.setChecked(config.ai_skip_if_cv_confident)
        ai_form.addRow(self._ai_skip)
        self._ai_interval = QDoubleSpinBox()
        self._ai_interval.setRange(1.0, 10.0)
        self._ai_interval.setValue(config.ai_live_interval_seconds)
        ai_form.addRow("Live analysis interval (s)", self._ai_interval)
        self._ollama_model = QLineEdit(config.ollama_model)
        self._ollama_model.setPlaceholderText("llama3.2")
        ai_form.addRow("Ollama model", self._ollama_model)

        self._ai_hint = QLabel("Requires Ollama: ollama pull llama3.2")
        self._ai_hint.setWordWrap(True)
        self._ai_hint.setStyleSheet("color: #666;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        test_btn = buttons.addButton("Test Ollama", QDialogButtonBox.ButtonRole.ActionRole)
        test_btn.clicked.connect(self._test_ollama)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Tune detection, ground presets, and AI advisory."))
        layout.addLayout(form)
        layout.addWidget(hsv_box)
        layout.addWidget(clip_box)
        layout.addWidget(ai_box)
        layout.addWidget(self._ai_hint)
        layout.addWidget(buttons)

    def _test_ollama(self) -> None:
        self.config.ai_enabled = self._ai_enabled.isChecked()
        self.config.ollama_model = self._ollama_model.text().strip() or "llama3.2"
        service = create_advisory_service(self.config)
        if service.provider_name == "none":
            QMessageBox.warning(self, "Ollama test", "Enable AI advisory first.")
            return
        if service.is_available():
            QMessageBox.information(
                self, "Ollama test",
                f"Connected.\nModel: {self.config.ollama_model}",
            )
        else:
            QMessageBox.warning(
                self, "Ollama test",
                "Cannot reach Ollama at http://127.0.0.1:11434\n\n"
                "Install from https://ollama.com and run: ollama pull llama3.2",
            )

    def _save(self) -> None:
        ground = self._ground.currentText().strip() or "default"
        if ground != self.config.ground_id:
            switch_ground(self.config, ground)
        else:
            self.config.ground_id = ground

        self.config.detection_mode = self._detection_mode.currentText()
        self.config.detection_scale = self._scale.value()
        self.config.pitch_area_min = self._pitch_min.value()
        self.config.batsman_area_min = self._batsman_min.value()
        self.config.ball_hsv = {k: spin.value() for k, spin in self._hsv_fields.items()}
        self.config.clip_pre_roll_seconds = self._pre_roll.value()
        self.config.clip_post_roll_seconds = self._post_roll.value()
        self.config.ring_buffer_seconds = self._buffer.value()
        self.config.ai_enabled = self._ai_enabled.isChecked()
        self.config.ai_live_enabled = self._ai_live.isChecked()
        self.config.ai_skip_if_cv_confident = self._ai_skip.isChecked()
        self.config.ai_live_interval_seconds = self._ai_interval.value()
        model = self._ollama_model.text().strip()
        if model:
            self.config.ollama_model = model
        save_ground_preset(self.config)
        save_config(self.config, user_config_path())
        self.accept()
