"""DRS Pro main window."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from drs import __app_name__, __version__
from drs.app.about_dialog import AboutDialog
from drs.app.playback_worker import PlaybackWorker
from drs.app.settings_dialog import SettingsDialog
from drs.app.setup_wizard import SetupWizard
from drs.app.timeline_widget import TimelineWidget
from drs.app.video_widget import VideoWidget
from drs.config import DRSConfig, save_config
from drs.engine import DRSEngine
from drs.paths import user_calibration_dir, user_config_path
from drs.ui.calibrate_stumps import calibrate_stumps_on_frame, read_calibration_frame, save_pitch_calibration
from drs.ui.playback import ViewMode

logger = logging.getLogger("drs.app")


class MainWindow(QMainWindow):
    def __init__(self, config: DRSConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle(__app_name__)
        self.resize(1280, 720)

        self._engine = DRSEngine(config)
        self._worker: PlaybackWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self._video = VideoWidget()
        self._timeline = TimelineWidget()
        layout.addWidget(self._video, stretch=1)
        layout.addWidget(self._timeline)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(f"{__app_name__} v{__version__} — Open a video to begin")

        self._build_menus()
        self._build_toolbar()
        self._bind_shortcuts()
        self._bind_timeline()

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_act = QAction("&Open Video...", self)
        open_act.triggered.connect(self._open_video)
        file_menu.addAction(open_act)
        save_act = QAction("&Save Clip", self)
        save_act.triggered.connect(self._save_clip)
        file_menu.addAction(save_act)
        file_menu.addSeparator()
        exit_act = QAction("E&xit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        tools_menu = self.menuBar().addMenu("&Tools")
        cal_act = QAction("&Calibrate Stumps...", self)
        cal_act.triggered.connect(self._calibrate)
        tools_menu.addAction(cal_act)
        setup_act = QAction("&Setup Wizard...", self)
        setup_act.triggered.connect(self._run_wizard)
        tools_menu.addAction(setup_act)
        settings_act = QAction("&Settings...", self)
        settings_act.triggered.connect(self._settings)
        tools_menu.addAction(settings_act)

        help_menu = self.menuBar().addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._about)
        help_menu.addAction(about_act)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        self.addToolBar(tb)
        for label, slot in (
            ("Open", self._open_video),
            ("Calibrate", self._calibrate),
            ("Save Clip", self._save_clip),
        ):
            act = QAction(label, self)
            act.triggered.connect(slot)
            tb.addAction(act)

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence("J"), self, lambda: self._step(1, back=True))
        QShortcut(QKeySequence("K"), self, lambda: self._step(1, back=False))
        QShortcut(QKeySequence("A"), self, lambda: self._step(5, back=True))
        QShortcut(QKeySequence("D"), self, lambda: self._step(5, back=False))
        QShortcut(QKeySequence("R"), self, self._restart)
        QShortcut(QKeySequence("S"), self, self._save_clip)

    def _bind_timeline(self) -> None:
        self._timeline.play_clicked.connect(self._play)
        self._timeline.pause_clicked.connect(self._pause)
        self._timeline.step_back_clicked.connect(lambda: self._step(5, back=True))
        self._timeline.step_forward_clicked.connect(lambda: self._step(5, back=False))
        self._timeline.restart_clicked.connect(self._restart)
        self._timeline.position_changed.connect(self._seek)

    def _stop_worker(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker = None

    def _start_worker(self) -> None:
        self._stop_worker()
        self._engine = DRSEngine(self.config)
        self._worker = PlaybackWorker(self._engine, self)
        self._worker.frame_ready.connect(self._video.show_frame)
        self._worker.position_changed.connect(self._on_position)
        self._worker.verdict_changed.connect(self._on_verdict)
        self._worker.playback_ended.connect(lambda: self._status.showMessage("End of video"))
        self._worker.error.connect(self._on_error)
        self._worker.fps_updated.connect(self._on_fps)
        self._timeline.set_total(self._engine.total_frames or 1)
        self._worker.start()

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video (*.mp4 *.avi *.mov)")
        if not path:
            return
        self.config.video_path = path
        save_config(self.config, user_config_path())
        self._start_worker()
        self._status.showMessage(f"Loaded: {Path(path).name}")

    def _play(self) -> None:
        if self._worker:
            self._worker.play()

    def _pause(self) -> None:
        if self._worker:
            self._worker.pause()

    def _toggle_play(self) -> None:
        if self._worker and self._worker.playback.mode == ViewMode.PLAYING:
            self._pause()
        else:
            self._play()

    def _step(self, delta: int, back: bool) -> None:
        if not self._worker:
            return
        if back:
            self._worker.step_back(delta)
        else:
            self._worker.step_forward(delta)

    def _restart(self) -> None:
        if self._worker:
            self._worker.restart()

    def _seek(self, pos: int) -> None:
        if self._worker:
            self._worker.seek(pos)

    def _save_clip(self) -> None:
        if not self._worker:
            return
        path = self._worker.engine.save_clip(self._worker.playback)
        if path:
            self._status.showMessage(f"Clip saved: {path}")
            QMessageBox.information(self, "Clip saved", str(path))
        else:
            QMessageBox.warning(self, "Save clip", "Nothing in buffer to save.")

    def _calibrate(self) -> None:
        video = self.config.video_path
        if not Path(video).is_file():
            QMessageBox.warning(self, "Calibrate", "Open a video first.")
            return
        frame = read_calibration_frame(video)
        if frame is None:
            QMessageBox.critical(self, "Calibrate", "Cannot read video frame.")
            return
        cam = calibrate_stumps_on_frame(frame, camera_name="primary")
        if cam is None:
            return
        out = user_calibration_dir() / f"{self.config.ground_id}.json"
        save_pitch_calibration(cam, ground_id=self.config.ground_id, output_path=out)
        self.config.calibration_file = str(out)
        save_config(self.config, user_config_path())
        self._engine.set_stump_points(cam.stump_points, from_calibration=True)
        QMessageBox.information(self, "Calibration saved", str(out))

    def _run_wizard(self) -> None:
        wizard = SetupWizard(self.config, self)
        if wizard.run_calibration():
            self._engine = DRSEngine(self.config)
            QMessageBox.information(self, "Setup complete", "Calibration saved. Open your video to begin.")

    def _settings(self) -> None:
        dlg = SettingsDialog(self.config, self)
        if dlg.exec():
            self._status.showMessage("Settings saved — reopen video to apply fully")

    def _about(self) -> None:
        AboutDialog(self).exec()

    def _on_position(self, pos: int, total: int) -> None:
        self._timeline.set_position(pos, total)
        self._timeline.set_total(total)

    def _on_verdict(self, verdict: str, reason: str) -> None:
        self._status.showMessage(f"Verdict: {verdict} — {reason}")
        logger.info("UI verdict: %s", verdict)

    def _on_error(self, msg: str) -> None:
        logger.error(msg)
        QMessageBox.critical(self, "Error", msg)

    def _on_fps(self, fps: float) -> None:
        if fps < 15:
            self._status.showMessage(f"Low FPS ({fps:.0f}) — lower detection scale in Settings")

    def closeEvent(self, event) -> None:
        self._stop_worker()
        event.accept()

    @staticmethod
    def needs_setup(config: DRSConfig) -> bool:
        if config.calibration_file and Path(config.calibration_file).is_file():
            return False
        user_cal = user_calibration_dir() / f"{config.ground_id}.json"
        return not user_cal.is_file()
