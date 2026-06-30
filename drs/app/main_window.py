"""DRS Pro main window."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QDockWidget,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from drs import __app_name__, __version__
from drs.app.about_dialog import AboutDialog
from drs.app.advisory_panel import AdvisoryPanel
from drs.app.advisory_worker import AdvisoryWorker
from drs.app.pitch_diagram_widget import PitchDiagramWidget
from drs.app.playback_worker import PlaybackWorker
from drs.app.settings_dialog import SettingsDialog
from drs.app.setup_wizard import SetupWizard
from drs.app.timeline_widget import TimelineWidget
from drs.app.video_widget import VideoWidget
from drs.config import DRSConfig, save_config
from drs.grounds import apply_ground_preset, save_ground_preset
from drs.app.clip_browser_dialog import ClipBrowserDialog
from drs.app.live_setup_dialog import LiveSetupDialog
from drs.engine import DRSEngine
from drs.paths import app_root, bundled_doc_path, user_calibration_dir, user_config_path
from drs.services.advisory.factory import create_advisory_service
from drs.services.advisory.models import AdvisoryResult, DeliveryEvidence
from drs.ui.calibrate_stumps import calibrate_stumps_on_frame, read_calibration_frame, save_pitch_calibration
from drs.ui.playback import ViewMode

logger = logging.getLogger("drs.app")


class MainWindow(QMainWindow):
    def __init__(self, config: DRSConfig, app_icon: QIcon | None = None):
        super().__init__()
        self.config = config
        self.setWindowTitle(__app_name__)
        if app_icon is not None and not app_icon.isNull():
            self.setWindowIcon(app_icon)
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

        self._diagram = PitchDiagramWidget()
        dock = QDockWidget("Pitch diagram", self)
        dock.setWidget(self._diagram)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self._advisory_service = create_advisory_service(config)
        self._advisory_worker = AdvisoryWorker(self._advisory_service, self)
        self._advisory_panel = AdvisoryPanel()
        self._advisory_dock = QDockWidget("AI Review", self)
        self._advisory_dock.setWidget(self._advisory_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._advisory_dock)
        if not config.ai_enabled:
            self._advisory_panel.set_enabled(False)
        else:
            self._advisory_worker.check_provider()
        self._advisory_worker.advisory_ready.connect(self._on_advisory_ready)
        self._advisory_worker.advisory_failed.connect(self._on_advisory_failed)
        self._advisory_worker.provider_status.connect(self._on_advisory_status)

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
        live_act = QAction("Start &Live Match...", self)
        live_act.triggered.connect(self._start_live)
        file_menu.addAction(live_act)
        save_act = QAction("&Save Clip", self)
        save_act.triggered.connect(self._save_clip)
        file_menu.addAction(save_act)
        clips_act = QAction("&Browse Clips...", self)
        clips_act.triggered.connect(self._browse_clips)
        file_menu.addAction(clips_act)
        file_menu.addSeparator()
        exit_act = QAction("E&xit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        tools_menu = self.menuBar().addMenu("&Tools")
        export_act = QAction("Export &Diagram Video...", self)
        export_act.triggered.connect(self._export_diagram)
        tools_menu.addAction(export_act)
        ai_act = QAction("Analyze Delivery with &AI", self)
        ai_act.triggered.connect(self._analyze_with_ai)
        tools_menu.addAction(ai_act)
        ai_settings_act = QAction("AI &Settings (Ollama)...", self)
        ai_settings_act.triggered.connect(self._settings)
        tools_menu.addAction(ai_settings_act)
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
        guide_act = QAction("&User Guide", self)
        guide_act.triggered.connect(self._open_user_guide)
        help_menu.addAction(guide_act)
        help_menu.addSeparator()
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._about)
        help_menu.addAction(about_act)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        self.addToolBar(tb)
        for label, slot in (
            ("Open", self._open_video),
            ("Live", self._start_live),
            ("DRS Call", self._drs_call),
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
        QShortcut(QKeySequence("F9"), self, self._drs_call)
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, self._jump_start)
        QShortcut(QKeySequence(Qt.Key.Key_End), self, self._jump_end)
        QShortcut(QKeySequence("B"), self, self._enter_buffer)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._exit_buffer)

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
        self._worker.playback_ended.connect(self._on_playback_ended)
        self._worker.error.connect(self._on_error)
        self._worker.fps_updated.connect(self._on_fps)
        self._worker.recording_status.connect(self._on_recording_status)
        self._worker.trajectory_updated.connect(self._on_trajectory)
        self._worker.drs_clip_ready.connect(self._on_drs_clip_ready)
        self._worker.delivery_evidence_ready.connect(self._on_delivery_evidence)
        self._refresh_advisory_service()
        self._worker.start()
        if not self.config.mode == "live":
            self._worker.play()
            self._status.showMessage("Playing — Space to pause, R to restart")

    def _refresh_advisory_service(self) -> None:
        self._advisory_service = create_advisory_service(self.config)
        self._advisory_worker.set_service(self._advisory_service)
        if self.config.ai_enabled:
            self._advisory_panel.set_enabled(True)
            self._advisory_dock.show()
            self._advisory_worker.check_provider()
        else:
            self._advisory_panel.set_enabled(False)
            self._advisory_panel.set_provider_status(False, "Disabled — enable in Settings")

    def _on_advisory_status(self, available: bool, label: str) -> None:
        self._advisory_panel.set_provider_status(available, label)
        if self.config.ai_enabled and not available:
            self._status.showMessage(
                "AI enabled but Ollama offline — run: ollama serve && ollama pull llama3.2"
            )

    def _on_playback_ended(self) -> None:
        self._status.showMessage("End of video — press Play or R to restart")
        if not self._worker or not self.config.diagram_enabled:
            return
        st = self._worker.engine.state
        if not st.trajectory_pitch_points:
            return
        self._diagram.set_trajectory(
            list(st.trajectory_pitch_points),
            pitch_bounce=self._worker._plane_point(st.pitch_point),
            impact=self._worker._plane_point(st.impact_point),
            pixel_points=[(p[0], p[1]) for p in st.trajectory_points],
            frame_h=self._worker._last_frame_h,
            stump_points=self._worker.engine.stump_points,
            animate=True,
        )

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video (*.mp4 *.avi *.mov)")
        if not path:
            return
        self._load_video_path(path)

    def _load_video_path(self, path: str) -> None:
        self.config.mode = "file"
        self.config.video_path = path
        save_config(self.config, user_config_path())
        self._start_worker()
        self._status.showMessage(f"Loaded: {Path(path).name}")

    def _start_live(self) -> None:
        dlg = LiveSetupDialog(self.config, self)
        if not dlg.exec():
            return
        save_ground_preset(self.config)
        self._start_worker()
        self._status.showMessage("Live match — recording started (see status bar)")
        if MainWindow.needs_setup(self.config):
            reply = QMessageBox.question(
                self,
                "Calibrate stumps",
                "No calibration for this ground.\n\nCalibrate stumps from the live camera now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._calibrate(from_live=True)

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

    def _drs_call(self) -> None:
        if not self._worker or not self._worker.engine.is_live:
            QMessageBox.information(self, "DRS Call", "Start a live match first (File → Start Live Match).")
            return
        self._worker.drs_call()
        self._status.showMessage(
            f"DRS call — saving clip (wait {self.config.clip_post_roll_seconds:.0f}s post-roll)..."
        )

    def _on_drs_clip_ready(self, clip_path: str) -> None:
        self._status.showMessage(f"DRS clip saved: {clip_path}")
        if self.config.ai_enabled and self._worker:
            self._worker.request_advisory(clip_path=clip_path)
        reply = QMessageBox.question(
            self,
            "DRS clip ready",
            f"Clip saved:\n{clip_path}\n\nOpen for frame-by-frame review?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._load_video_path(clip_path)

    def _on_recording_status(self, elapsed: float, path: str) -> None:
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        name = Path(path).name if path else "starting..."
        self._status.showMessage(f"REC {mins:02d}:{secs:02d} — {name}")

    def _on_trajectory(self, payload) -> None:
        if not self.config.diagram_enabled:
            return
        if isinstance(payload, dict):
            points = payload.get("points", [])
            pitch_pt = payload.get("pitch_bounce")
            impact_pt = payload.get("impact")
            pixel_pts = payload.get("pixel_points")
            frame_h = payload.get("frame_h", 720)
            stump_points = payload.get("stump_points")
            animate = payload.get("animate", False)
            live_ball = payload.get("live_ball")
            live_pixel = payload.get("live_pixel")
        else:
            points = list(payload)
            pitch_pt = None
            impact_pt = None
            pixel_pts = None
            frame_h = 720
            stump_points = None
            animate = False
            live_ball = None
            live_pixel = None
        if self._worker and not pitch_pt and self._worker.engine.state.pitch_point:
            pitch_pt = self._worker.engine.state.trajectory_pitch_points[-1] if self._worker.engine.state.trajectory_pitch_points else None
        self._diagram.set_trajectory(
            points,
            pitch_bounce=pitch_pt,
            impact=impact_pt,
            pixel_points=pixel_pts,
            frame_h=frame_h,
            stump_points=stump_points,
            animate=animate,
            live_ball=live_ball,
            live_pixel=live_pixel,
        )

    def _export_diagram(self) -> None:
        video = self.config.video_path
        if not Path(video).is_file():
            QMessageBox.warning(self, "Export diagram", "Open a video first.")
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "Save diagram video", f"{Path(video).stem}_diagram.mp4", "Video (*.mp4)",
        )
        if not out:
            return
        try:
            from drs.analysis.video_trajectory import extract_trajectories_from_video
            from drs.ui.pitch_diagram import export_combined_diagram_video

            deliveries = extract_trajectories_from_video(video, self.config)
            if not deliveries:
                QMessageBox.warning(self, "Export diagram", "No ball trajectory detected in this video.")
                return
            best = max(deliveries, key=lambda d: len(d.pitch_points))
            export_combined_diagram_video(
                best.pitch_points,
                best.pixel_points,
                out,
                fps=24.0,
                frame_h=best.frame_h,
                pitch_bounce=best.pitch_bounce,
                impact=best.impact,
                stump_points=best.stump_points,
            )
            self._diagram.set_trajectory(
                best.pitch_points,
                pitch_bounce=best.pitch_bounce,
                impact=best.impact,
                pixel_points=best.pixel_points,
                frame_h=best.frame_h,
                stump_points=best.stump_points,
                animate=True,
            )
            QMessageBox.information(
                self, "Diagram exported",
                f"Saved: {out}\n\n{best.pitch_points.__len__()} tracking points from delivery #{best.delivery_index}.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export diagram", str(exc))

    def _save_clip(self) -> None:
        if not self._worker:
            return
        path = self._worker.engine.save_clip(self._worker.playback)
        if path:
            self._status.showMessage(f"Clip saved: {path}")
            QMessageBox.information(self, "Clip saved", str(path))
        else:
            QMessageBox.warning(self, "Save clip", "Nothing in buffer to save.")

    def _calibrate(self, *, from_live: bool = False) -> None:
        frame = None
        if from_live or (self._worker and self._worker.engine.is_live):
            if not self._worker:
                QMessageBox.warning(self, "Calibrate", "Start live match first.")
                return
            frame = self._worker.capture_calibration_frame()
            if frame is None:
                QMessageBox.warning(self, "Calibrate", "Cannot read live camera frame.")
                return
        else:
            video = self.config.video_path
            if not Path(video).is_file():
                QMessageBox.warning(self, "Calibrate", "Open a video or start live match first.")
                return
            frame = read_calibration_frame(video)
        if frame is None:
            QMessageBox.critical(self, "Calibrate", "Cannot read calibration frame.")
            return
        cam = calibrate_stumps_on_frame(frame, camera_name="primary")
        if cam is None:
            return
        out = user_calibration_dir() / f"{self.config.ground_id}.json"
        save_pitch_calibration(cam, ground_id=self.config.ground_id, output_path=out)
        self.config.calibration_file = str(out)
        save_config(self.config, user_config_path())
        save_ground_preset(self.config)
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
            was_running = self._worker is not None
            self._refresh_advisory_service()
            if was_running:
                self._start_worker()
            msg = "Settings saved."
            if self.config.ai_enabled:
                msg += " AI Review enabled — use Tools → Analyze Delivery with AI after a pad/LBW moment."
            else:
                msg += " Reopen video to apply detection changes fully."
            self._status.showMessage(msg)

    def _about(self) -> None:
        AboutDialog(self).exec()

    def _on_position(self, pos: int, total: int) -> None:
        self._timeline.set_position(pos, total)
        self._timeline.set_total(total)

    def _on_verdict(self, verdict: str, reason: str) -> None:
        conf_pct = 0
        ai_note = ""
        ai_verdict = ""
        if self._worker:
            conf_pct = int(self._worker.engine.state.confidence_overall * 100)
            if self._worker.engine.state.ai_verdict:
                ai_verdict = self._worker.engine.state.ai_verdict
                ai_note = f" — AI: {ai_verdict}"
        self._video.set_verdict_banner(verdict, confidence=conf_pct, ai_verdict=ai_verdict)
        self._status.showMessage(f"Verdict: {verdict} ({conf_pct}% confidence){ai_note} — {reason}")
        logger.info("UI verdict: %s (%d%%)", verdict, conf_pct)

    def _on_delivery_evidence(self, payload) -> None:
        if not self.config.ai_enabled:
            return
        if isinstance(payload, dict):
            evidence = payload.get("evidence")
            force = bool(payload.get("force", False))
        else:
            evidence = payload
            force = False
        if evidence is None:
            return
        self._advisory_panel.show_evidence(evidence)
        self._advisory_panel.set_analyzing()
        self._advisory_worker.analyze(evidence, force=force)

    def _on_advisory_ready(self, result: AdvisoryResult) -> None:
        self._advisory_panel.show_result(result)
        if self._worker:
            self._worker.engine.apply_advisory_result(result)
            self._worker.engine.state.ai_pending = False
        ai = result.recommended_verdict
        self._status.showMessage(
            f"AI review: {ai} ({int(result.confidence * 100)}%) — {result.summary}"
        )

    def _on_advisory_failed(self, message: str) -> None:
        self._advisory_panel.show_error(message)
        if self._worker:
            self._worker.engine.state.ai_pending = False
            self._worker._ai_pending_since = 0.0

    def _analyze_with_ai(self) -> None:
        if not self.config.ai_enabled:
            reply = QMessageBox.question(
                self,
                "AI Advisory",
                "AI Review is not enabled.\n\nOpen Settings now to enable Ollama advisory?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._settings()
            return
        if not self._worker:
            QMessageBox.warning(self, "AI Advisory", "Open a video or start live match first.")
            return
        if self._advisory_service.provider_name == "none":
            QMessageBox.warning(self, "AI Advisory", "AI service not configured.")
            return
        if not self._advisory_service.is_available():
            QMessageBox.warning(
                self,
                "Ollama offline",
                "Cannot reach Ollama at http://127.0.0.1:11434\n\n"
                "1. Install Ollama from https://ollama.com\n"
                "2. In a terminal: ollama pull llama3.2\n"
                "3. Ensure Ollama is running (system tray)\n"
                "4. Tools → Settings → Test Ollama",
            )
            return
        ok = self._worker.request_advisory()
        if not ok:
            QMessageBox.information(
                self,
                "AI Advisory",
                "Could not build delivery evidence. Play or scrub to the LBW/pad moment "
                "(when OUT/NOT OUT/REVIEW appears), then try again.",
            )
            return
        self._status.showMessage("AI analysis requested — see AI Review panel…")

    def _on_error(self, msg: str) -> None:
        logger.error(msg)
        QMessageBox.critical(self, "Error", msg)

    def _on_fps(self, fps: float) -> None:
        if fps < 15:
            self._status.showMessage(f"Low FPS ({fps:.0f}) — lower detection scale in Settings")

    def closeEvent(self, event) -> None:
        self._stop_worker()
        if self._advisory_worker.isRunning():
            self._advisory_worker.wait(3000)
        event.accept()

    def _browse_clips(self) -> None:
        ClipBrowserDialog(self, on_open_video=self._load_video_path).exec()

    def _open_user_guide(self) -> None:
        import os
        import subprocess
        import sys

        guide = bundled_doc_path("USER_GUIDE.md")
        dev_guide = app_root().parent / "docs" / "USER_GUIDE.md" if not guide.is_file() else guide
        path = guide if guide.is_file() else dev_guide
        if not path.is_file():
            QMessageBox.warning(self, "User Guide", "USER_GUIDE.md not found in installation.")
            return
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def _jump_start(self) -> None:
        if self._worker:
            self._worker.jump_start()

    def _jump_end(self) -> None:
        if self._worker:
            self._worker.jump_end()

    def _enter_buffer(self) -> None:
        if self._worker:
            self._worker.enter_buffer()

    def _exit_buffer(self) -> None:
        if self._worker:
            self._worker.exit_buffer()

    @staticmethod
    def needs_setup(config: DRSConfig) -> bool:
        if config.calibration_file and Path(config.calibration_file).is_file():
            return False
        user_cal = user_calibration_dir() / f"{config.ground_id}.json"
        return not user_cal.is_file()
