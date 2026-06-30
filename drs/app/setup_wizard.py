"""First-run setup wizard."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from drs.config import DRSConfig, resolve_calibration_path, save_config
from drs.paths import user_config_path
from drs.ui.calibrate_stumps import calibrate_stumps_on_frame, read_calibration_frame, save_pitch_calibration


class WelcomePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to DRS Pro")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "First-run checklist:\n\n"
            "• Stump calibration is required for accurate line assist\n"
            "• Sample video from your ground camera angle (or calibrate live later)\n"
            "• Ollama is optional — enable AI Review in Settings after install\n\n"
            "This wizard saves calibration per ground ID."
        ))


class GroundPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Ground profile")
        layout = QVBoxLayout(self)
        self._ground = QLineEdit("default")
        layout.addWidget(QLabel("Ground name (used for saved calibration):"))
        layout.addWidget(self._ground)

    def ground_id(self) -> str:
        return self._ground.text().strip() or "default"


class VideoPage(QWizardPage):
    def __init__(self, config: DRSConfig):
        super().__init__()
        self.config = config
        self.setTitle("Sample video")
        layout = QVBoxLayout(self)
        self._path = QLineEdit(config.video_path)
        btn = QLabel('<a href="#">Browse for video file...</a>')
        btn.setOpenExternalLinks(False)
        btn.linkActivated.connect(self._browse)
        layout.addWidget(QLabel("Choose a video from this ground (camera angle):"))
        layout.addWidget(self._path)
        layout.addWidget(btn)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video (*.mp4 *.avi *.mov)")
        if path:
            self._path.setText(path)

    def video_path(self) -> str:
        return self._path.text().strip()


class SetupWizard(QWizard):
    def __init__(self, config: DRSConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("DRS Pro Setup")
        self._welcome = WelcomePage()
        self._ground = GroundPage()
        self._video = VideoPage(config)
        self.addPage(self._welcome)
        self.addPage(self._ground)
        self.addPage(self._video)

    def run_calibration(self) -> bool:
        if self.exec() != QWizard.DialogCode.Accepted:
            return False

        ground_id = self._ground.ground_id()
        video = self._video.video_path()
        frame = read_calibration_frame(video)
        if frame is None:
            return False

        cam = calibrate_stumps_on_frame(frame, camera_name="primary")
        if cam is None:
            return False

        out = Path(resolve_calibration_path(ground_id))
        save_pitch_calibration(cam, ground_id=ground_id, output_path=out)

        self.config.ground_id = ground_id
        self.config.calibration_file = str(out)
        self.config.video_path = video
        save_config(self.config, user_config_path())
        return True
