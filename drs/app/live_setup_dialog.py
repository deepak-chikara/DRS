"""Live match setup dialog — camera and detection without editing YAML."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from drs.config import DRSConfig, save_config
from drs.grounds import apply_ground_preset, list_ground_ids, save_ground_preset
from drs.paths import user_config_path, user_matches_dir


class LiveSetupDialog(QDialog):
    def __init__(self, config: DRSConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Live Match Setup")
        self.setMinimumWidth(460)

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
        form.addRow("Ground ID", self._ground)

        self._camera_type = QComboBox()
        self._camera_type.addItems(["USB webcam", "RTSP / IP camera"])
        primary = config.cameras.get("primary")
        if primary and primary.type == "rtsp":
            self._camera_type.setCurrentIndex(1)
        form.addRow("Camera type", self._camera_type)

        self._source = QLineEdit()
        if primary:
            self._source.setText(str(primary.source))
        else:
            self._source.setText("0")
        self._source.setPlaceholderText("0 for first USB camera, or rtsp://...")
        form.addRow("Camera source", self._source)

        self._detection = QComboBox()
        self._detection.addItems(["hybrid", "color", "yolo"])
        idx = self._detection.findText(config.detection_mode)
        if idx >= 0:
            self._detection.setCurrentIndex(idx)
        form.addRow("Detection mode", self._detection)

        self._scale = QDoubleSpinBox()
        self._scale.setRange(0.25, 1.0)
        self._scale.setSingleStep(0.05)
        self._scale.setValue(min(config.detection_scale, 0.5) if config.detection_scale > 0.5 else config.detection_scale)
        form.addRow("Detection scale (live)", self._scale)

        self._buffer = QSpinBox()
        self._buffer.setRange(15, 120)
        self._buffer.setValue(config.ring_buffer_seconds)
        form.addRow("Ring buffer (seconds)", self._buffer)

        self._record = QCheckBox("Record full match segments")
        self._record.setChecked(config.recording_enabled)
        form.addRow(self._record)

        self._rec_width = QSpinBox()
        self._rec_width.setRange(640, 3840)
        self._rec_width.setValue(config.recording_width or 1280)
        form.addRow("Recording width", self._rec_width)

        self._pre_roll = QDoubleSpinBox()
        self._pre_roll.setRange(2, 30)
        self._pre_roll.setValue(config.clip_pre_roll_seconds)
        form.addRow("DRS clip pre-roll (s)", self._pre_roll)

        self._post_roll = QDoubleSpinBox()
        self._post_roll.setRange(2, 30)
        self._post_roll.setValue(config.clip_post_roll_seconds)
        form.addRow("DRS clip post-roll (s)", self._post_roll)

        hint = QLabel(
            "After starting live match, use Tools → Calibrate Stumps to click stump bases "
            "on the current camera frame."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Configure live match</b> — no YAML editing required."))
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)

    def _apply(self) -> None:
        ground = self._ground.currentText().strip() or "default"
        apply_ground_preset(self.config, ground)

        self.config.mode = "live"
        self.config.ground_id = ground
        is_rtsp = self._camera_type.currentIndex() == 1
        source_text = self._source.text().strip()
        if is_rtsp:
            source: str | int = source_text
            cam_type = "rtsp"
        else:
            try:
                source = int(source_text)
            except ValueError:
                source = 0
            cam_type = "usb"

        from drs.config import CameraConfig

        self.config.cameras = {
            "primary": CameraConfig("primary", cam_type, source, True),
        }
        self.config.detection_mode = self._detection.currentText()
        self.config.detection_scale = self._scale.value()
        self.config.ring_buffer_seconds = self._buffer.value()
        self.config.recording_enabled = self._record.isChecked()
        if not self.config.recording_output_dir:
            self.config.recording_output_dir = str(user_matches_dir())
        self.config.recording_width = self._rec_width.value()
        self.config.clip_pre_roll_seconds = self._pre_roll.value()
        self.config.clip_post_roll_seconds = self._post_roll.value()

        save_ground_preset(self.config)
        save_config(self.config, user_config_path())
        self.accept()
