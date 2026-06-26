"""Background playback thread."""

from __future__ import annotations

import time

from PySide6.QtCore import QMutex, QThread, Signal

from drs.engine import DRSEngine
from drs.ui.playback import PlaybackState, ViewMode


class PlaybackWorker(QThread):
    frame_ready = Signal(object)
    position_changed = Signal(int, int)
    verdict_changed = Signal(str, str)
    playback_ended = Signal()
    error = Signal(str)
    fps_updated = Signal(float)

    def __init__(self, engine: DRSEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.playback = PlaybackState()
        self._running = True
        self._mutex = QMutex()
        self._total = 0
        self._command: str | None = None

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def _set_command(self, cmd: str) -> None:
        self._mutex.lock()
        self._command = cmd
        self._mutex.unlock()

    def play(self) -> None:
        self._set_command("play")

    def pause(self) -> None:
        self._set_command("pause")

    def step_back(self, delta: int = 5) -> None:
        self._mutex.lock()
        self._command = f"back:{delta}"
        self._mutex.unlock()

    def step_forward(self, delta: int = 5) -> None:
        self._mutex.lock()
        self._command = f"fwd:{delta}"
        self._mutex.unlock()

    def restart(self) -> None:
        self._set_command("restart")

    def seek(self, frame_pos: int) -> None:
        self._mutex.lock()
        self._command = f"seek:{frame_pos}"
        self._mutex.unlock()

    def _apply_command(self) -> None:
        self._mutex.lock()
        cmd = self._command
        self._command = None
        prev = self.playback.mode
        self._mutex.unlock()
        if not cmd:
            return

        if cmd == "play":
            self.playback.mode = ViewMode.PLAYING
            self.engine.on_playback_action("pause", self.playback, ViewMode.PAUSED)
        elif cmd == "pause":
            self.playback.mode = ViewMode.PAUSED
            self.engine.on_playback_action("pause", self.playback, ViewMode.PLAYING)
        elif cmd == "restart":
            self.engine.on_playback_action("restart", self.playback, prev)
            self.playback.mode = ViewMode.PAUSED
        elif cmd.startswith("seek:"):
            self.playback.frame_pos = int(cmd.split(":")[1])
            self.playback.mode = ViewMode.PAUSED
            self.engine._sequential_play = False
        elif cmd.startswith("back:"):
            delta = int(cmd.split(":")[1])
            self.playback.frame_pos = max(0, self.playback.frame_pos - delta)
            self.playback.mode = ViewMode.PAUSED
            self.engine._sequential_play = False
        elif cmd.startswith("fwd:"):
            delta = int(cmd.split(":")[1])
            max_pos = max(0, self._total - 1)
            self.playback.frame_pos = min(max_pos, self.playback.frame_pos + delta)
            self.playback.mode = ViewMode.PAUSED
            self.engine._sequential_play = False

    def run(self) -> None:
        ok, err = self.engine.open()
        if not ok:
            self.error.emit(err)
            return

        self._total = self.engine.total_frames
        self.position_changed.emit(self.playback.frame_pos, self._total)
        last_verdict = ""

        while self._running:
            self._apply_command()
            pb = self.playback

            frame, read_ok = self.engine.read_frame(pb)
            if not read_ok:
                if pb.mode == ViewMode.PLAYING:
                    self.playback_ended.emit()
                    self.playback.mode = ViewMode.PAUSED
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            self.engine.notify_frame_position(pb.frame_pos)
            display = self.engine.process_frame(frame)
            self.engine.push_buffer(frame, display)
            self.frame_ready.emit(display)

            if pb.mode == ViewMode.PLAYING:
                pos = self.playback.frame_pos
                self.playback.frame_pos += 1
                self.position_changed.emit(pos, self._total)
            else:
                self.position_changed.emit(pb.frame_pos, self._total)

            if self.engine.state.verdict and self.engine.state.verdict != last_verdict:
                last_verdict = self.engine.state.verdict
                self.verdict_changed.emit(self.engine.state.verdict, self.engine.state.verdict_reason)

            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                self.fps_updated.emit(1.0 / elapsed)

            if pb.mode == ViewMode.PLAYING and self.engine.video_fps > 0:
                delay = max(0.001, 1.0 / self.engine.video_fps - elapsed)
                time.sleep(delay)
            else:
                time.sleep(0.03)

        self.engine.close()
