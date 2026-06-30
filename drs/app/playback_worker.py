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
    recording_status = Signal(float, str)
    trajectory_updated = Signal(object)
    drs_clip_ready = Signal(str)
    delivery_evidence_ready = Signal(object)

    def __init__(self, engine: DRSEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.playback = PlaybackState()
        self._running = True
        self._mutex = QMutex()
        self._total = 0
        self._command: str | None = None
        self._last_frame_h = 720
        self._last_frame_w = 1280
        self._at_eof = False
        self._last_trajectory_len = 0
        self._last_processed_pos = -1
        self._frames_seen = 0
        self._last_ai_emit_time = 0.0
        self._last_ai_traj_count = 0
        self._ai_bootstrapped = False
        self._ai_pending_since = 0.0

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

    def drs_call(self) -> None:
        self._set_command("drs_call")

    def jump_start(self) -> None:
        self._set_command("jump_start")

    def jump_end(self) -> None:
        self._set_command("jump_end")

    def enter_buffer(self) -> None:
        self._set_command("buffer_enter")

    def exit_buffer(self) -> None:
        self._set_command("buffer_exit")

    def capture_calibration_frame(self):
        import numpy as np

        if not self.engine.is_live:
            return None
        pkt = self.engine._grabber.read_latest() if self.engine._grabber else None
        if pkt is None or pkt.frame is None:
            return None
        return pkt.frame.copy()

    def _clear_eof(self) -> None:
        self._at_eof = False

    def _update_total(self, frame_pos: int) -> None:
        observed = max(frame_pos + 1, self._frames_seen)
        if observed > self._total:
            self._total = observed

    def _apply_command(self) -> None:
        self._mutex.lock()
        cmd = self._command
        self._command = None
        prev = self.playback.mode
        self._mutex.unlock()
        if not cmd:
            return

        if cmd == "play":
            if self._at_eof:
                self._clear_eof()
                self.engine.on_playback_action("restart", self.playback, ViewMode.PAUSED)
                self.playback.frame_pos = 0
                self._last_processed_pos = -1
            else:
                self._clear_eof()
            self.playback.mode = ViewMode.PLAYING
            self.engine.on_playback_action("pause", self.playback, ViewMode.PAUSED)
        elif cmd == "pause":
            self.playback.mode = ViewMode.PAUSED
            self.engine.on_playback_action("pause", self.playback, ViewMode.PLAYING)
        elif cmd == "restart":
            self._clear_eof()
            self._last_processed_pos = -1
            self._ai_bootstrapped = False
            self._last_ai_emit_time = 0.0
            self.engine.on_playback_action("restart", self.playback, prev)
            self.playback.mode = ViewMode.PAUSED
        elif cmd == "drs_call":
            self.engine.trigger_drs_call()
        elif cmd == "jump_start":
            self._clear_eof()
            self.playback.frame_pos = 0
            self.playback.mode = ViewMode.PAUSED
            self.engine.on_playback_action("jump_start", self.playback, prev)
            self._last_processed_pos = -1
        elif cmd == "jump_end":
            self._clear_eof()
            max_pos = max(0, self._total - 1)
            self.playback.frame_pos = max_pos
            self.playback.mode = ViewMode.PAUSED
            self.engine.on_playback_action("jump_end", self.playback, prev)
            self._last_processed_pos = -1
        elif cmd == "buffer_enter":
            buf_len = len(self.engine.ring_buffer)
            if buf_len > 0:
                self.playback.mode = ViewMode.BUFFER
                self.playback.buffer_index = buf_len - 1
                self._last_processed_pos = -1
        elif cmd == "buffer_exit":
            if self.playback.mode == ViewMode.BUFFER:
                self.playback.mode = ViewMode.PAUSED
                self._last_processed_pos = -1
        elif cmd.startswith("seek:"):
            self._clear_eof()
            max_pos = max(0, self._total - 1)
            self.playback.frame_pos = min(max_pos, int(cmd.split(":")[1]))
            self.playback.mode = ViewMode.PAUSED
            self.engine._sequential_play = False
            self._last_processed_pos = -1
        elif cmd.startswith("back:"):
            self._clear_eof()
            delta = int(cmd.split(":")[1])
            if self.playback.mode == ViewMode.BUFFER:
                buf_len = len(self.engine.ring_buffer)
                if buf_len:
                    self.playback.buffer_index = max(0, self.playback.buffer_index - delta)
            else:
                self.playback.frame_pos = max(0, self.playback.frame_pos - delta)
            self.playback.mode = ViewMode.PAUSED if self.playback.mode != ViewMode.BUFFER else ViewMode.BUFFER
            self.engine._sequential_play = False
            self._last_processed_pos = -1
        elif cmd.startswith("fwd:"):
            self._clear_eof()
            delta = int(cmd.split(":")[1])
            if self.playback.mode == ViewMode.BUFFER:
                buf_len = len(self.engine.ring_buffer)
                if buf_len:
                    self.playback.buffer_index = min(buf_len - 1, self.playback.buffer_index + delta)
            else:
                max_pos = max(0, self._total - 1)
                self.playback.frame_pos = min(max_pos, self.playback.frame_pos + delta)
            self.playback.mode = ViewMode.PAUSED if self.playback.mode != ViewMode.BUFFER else ViewMode.BUFFER
            self.engine._sequential_play = False
            self._last_processed_pos = -1

    def _emit_last_frame(self) -> None:
        """Re-display the last valid frame after hitting EOF."""
        if self.engine.is_live:
            return
        last_pos = max(0, self._total - 1, self._frames_seen - 1)
        self._total = last_pos + 1
        self.playback.frame_pos = last_pos
        self.engine._sequential_play = False
        self._last_processed_pos = -1
        frame, read_ok = self.engine.read_frame(self.playback)
        if not read_ok or frame is None:
            return
        display = self.engine.process_frame(frame)
        self.engine.push_buffer(frame, display)
        self.frame_ready.emit(display)
        self._last_processed_pos = last_pos
        self.position_changed.emit(last_pos, self._total)

    def run(self) -> None:
        ok, err = self.engine.open()
        if not ok:
            self.error.emit(err)
            return

        self._total = max(self.engine.total_frames, 1)
        self._frames_seen = 0
        self.position_changed.emit(self.playback.frame_pos, self._total)
        last_verdict = ""

        while self._running:
            self._apply_command()
            pb = self.playback

            if self._at_eof and pb.mode != ViewMode.PLAYING:
                time.sleep(0.05)
                continue

            if (
                pb.mode != ViewMode.PLAYING
                and pb.frame_pos == self._last_processed_pos
            ):
                time.sleep(0.05)
                continue

            frame, read_ok = self.engine.read_frame(pb)
            if not read_ok:
                if pb.mode == ViewMode.PLAYING:
                    self._at_eof = True
                    self._total = max(self._total, self._frames_seen, pb.frame_pos)
                    self.playback.frame_pos = max(0, self._total - 1)
                    self.playback.mode = ViewMode.PAUSED
                    self.engine._sequential_play = False
                    self._emit_last_frame()
                    self._emit_trajectory(final=True)
                    self.playback_ended.emit()
                elif pb.frame_pos > 0:
                    self.playback.frame_pos = min(pb.frame_pos, max(0, self._total - 1))
                    self._last_processed_pos = -1
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()
            self._last_frame_h = frame.shape[0]
            self._last_frame_w = frame.shape[1]
            self._frames_seen = max(self._frames_seen, pb.frame_pos + 1)
            self._update_total(pb.frame_pos)
            if not self.engine.is_live:
                self.engine.notify_frame_position(pb.frame_pos)
            display = self.engine.process_frame(frame)
            self.engine.push_buffer(frame, display)
            self.frame_ready.emit(display)
            self._last_processed_pos = pb.frame_pos

            if self.engine.is_live:
                elapsed = self.engine.recording_elapsed
                rec_path = str(self.engine.match_recording_path or "")
                self.recording_status.emit(elapsed, rec_path)
                clip_path = self.engine.finalize_drs_call_if_ready()
                if clip_path:
                    self.drs_clip_ready.emit(str(clip_path))

            if self.engine.config.diagram_enabled:
                self._emit_trajectory(
                    final=False,
                    live=pb.mode == ViewMode.PLAYING and not self._at_eof,
                )

            if self.engine.config.ai_enabled:
                self._maybe_emit_live_advisory()

            if pb.mode == ViewMode.PLAYING:
                pos = self.playback.frame_pos
                self.playback.frame_pos += 1
                self.position_changed.emit(pos, self._total)
            else:
                self.position_changed.emit(pb.frame_pos, self._total)

            if self.engine.state.verdict and self.engine.state.verdict != last_verdict:
                last_verdict = self.engine.state.verdict
                self.verdict_changed.emit(self.engine.state.verdict, self.engine.state.verdict_reason)
                self._emit_delivery_evidence()

            elapsed_proc = time.perf_counter() - t0
            if elapsed_proc > 0:
                self.fps_updated.emit(1.0 / elapsed_proc)

            if pb.mode == ViewMode.PLAYING and self.engine.video_fps > 0:
                delay = max(0.001, 1.0 / self.engine.video_fps - elapsed_proc)
                time.sleep(delay)
            elif pb.mode == ViewMode.PLAYING:
                time.sleep(0.03)

        self.engine.close()

    def _emit_trajectory(self, *, final: bool = False, live: bool = False) -> None:
        st = self.engine.state
        points = list(st.trajectory_pitch_points)
        live_ball = self._live_ball_pitch() if live else None
        live_pixel = None
        if live_ball is not None and st.ball.x and st.ball.y:
            live_pixel = (float(st.ball.x), float(st.ball.y))

        if not final and not live and len(points) == self._last_trajectory_len:
            return
        if not final and live and not points and live_ball is None:
            return
        self._last_trajectory_len = len(points)
        payload = {
            "points": points,
            "pixel_points": [(p[0], p[1]) for p in st.trajectory_points],
            "frame_h": self._last_frame_h,
            "stump_points": self.engine.stump_points,
            "pitch_bounce": self._plane_point(st.pitch_point),
            "impact": self._plane_point(st.impact_point),
            "verdict": st.verdict,
            "animate": final,
            "live_ball": live_ball,
            "live_pixel": live_pixel,
        }
        self.trajectory_updated.emit(payload)

    def _live_ball_pitch(self) -> tuple[float, float] | None:
        st = self.engine.state
        if st.ball.x == 0 and st.ball.y == 0:
            return None
        if not self.engine.event_detector._delivery_in_progress(st):
            return None
        return self._pixel_to_pitch(st.ball.x, st.ball.y, self._last_frame_w, self._last_frame_h)

    def _pixel_to_pitch(self, x: int, y: int, frame_w: int, frame_h: int) -> tuple[float, float]:
        from drs.fusion.calibration import pixel_to_pitch_normalized

        return pixel_to_pitch_normalized(
            x, y,
            frame_w=frame_w,
            frame_h=frame_h,
            homography=self.engine._pitch_homography,
            stump_points=self.engine.stump_points,
        )

    def _plane_point(self, pt: tuple[int, int] | None) -> tuple[float, float] | None:
        if pt is None:
            return None
        return self._pixel_to_pitch(pt[0], pt[1], self._last_frame_w, self._last_frame_h)

    def _emit_delivery_evidence(
        self,
        clip_path: str | None = None,
        *,
        manual: bool = False,
        live: bool = False,
    ) -> bool:
        if not self.engine.config.ai_enabled:
            return False
        from drs.services.advisory.evidence import build_delivery_evidence

        st = self.engine.state
        evidence = build_delivery_evidence(
            st,
            mode=self.engine.config.mode,
            frame_pos=self.playback.frame_pos,
            delivery_id=st.delivery_count or 1,
            stump_points=self.engine.stump_points,
            clip_path=clip_path,
            manual=manual,
            live=live,
        )
        if evidence is None:
            return False
        st.ai_pending = True
        self.delivery_evidence_ready.emit({
            "evidence": evidence,
            "force": manual or live,
        })
        return True

    def _maybe_emit_live_advisory(self) -> None:
        cfg = self.engine.config
        if not cfg.ai_enabled or not cfg.ai_live_enabled:
            return

        st = self.engine.state
        now = time.perf_counter()
        if cfg.ai_skip_if_cv_confident:
            if (
                st.verdict in ("OUT", "NOT OUT")
                and st.confidence_overall >= cfg.ai_cv_confidence_skip_threshold
            ):
                return

        if st.ai_pending:
            if self._ai_pending_since and (now - self._ai_pending_since) > 35.0:
                st.ai_pending = False
                self._ai_pending_since = 0.0
            else:
                if not self._ai_pending_since:
                    self._ai_pending_since = now
                return
        interval = max(1.0, cfg.ai_live_interval_seconds)
        ball_visible = st.ball.x != 0 or st.ball.y != 0
        traj = len(st.trajectory_pitch_points)
        stumps_ready = (
            self.engine.stump_points is not None
            and self.engine.stump_points.is_valid()
        )

        if not ball_visible and traj == 0 and not stumps_ready:
            return

        should_emit = False
        if not self._ai_bootstrapped and (ball_visible or stumps_ready):
            should_emit = True
            self._ai_bootstrapped = True
        elif ball_visible and (now - self._last_ai_emit_time) >= interval:
            should_emit = True
        elif self.playback.mode == ViewMode.PLAYING and (now - self._last_ai_emit_time) >= interval * 2:
            # Keep AI updating during playback even when ball flickers off-screen
            should_emit = True
        elif traj >= self._last_ai_traj_count + 2 and ball_visible:
            should_emit = True

        if should_emit and self._emit_delivery_evidence(live=True):
            self._last_ai_emit_time = now
            self._last_ai_traj_count = traj
            self._ai_pending_since = now

    def request_advisory(self, clip_path: str | None = None) -> bool:
        """On-demand AI analysis for current delivery."""
        return self._emit_delivery_evidence(clip_path=clip_path, manual=True)
