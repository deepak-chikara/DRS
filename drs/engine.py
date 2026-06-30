"""Headless DRS engine for Qt UI and programmatic use."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from drs.config import DRSConfig
from drs.decision.events import EventDetector
from drs.decision.verdict import VerdictEngine
from drs.detectors.hybrid import HybridDetector
from drs.fusion.calibration import PitchCalibration, StumpPoints, pixel_to_pitch_normalized
from drs.paths import app_root, user_calibration_dir, user_clips_dir, user_matches_dir, user_sessions_dir
from drs.recording.delivery_clip import DRSCallRecord, DeliveryClipExporter, export_drs_call_clip
from drs.recording.match_recorder import MatchRecorder
from drs.session.logger import ClipExporter, DeliveryRecord, SessionLog
from drs.sources import ThreadedFrameGrabber, VideoSource
from drs.state import DRSState
from drs.sync.frame_sync import BufferedFrame, RingBuffer
from drs.ui.display import draw_overlays, render_frame
from drs.ui.playback import PlaybackState, ViewMode

logger = logging.getLogger("drs.engine")


class DRSEngine:
    """Core detection and playback logic without OpenCV windowing."""

    def __init__(self, config: DRSConfig):
        self.config = config
        self.state = DRSState()
        self.event_detector = EventDetector(config)
        self.verdict_engine = VerdictEngine(config)
        self.ring_buffer = RingBuffer(config.ring_buffer_capacity)
        self.session_log = SessionLog(ground_name=config.ground_id or "DRS Session")
        clips = user_clips_dir() if config.session_log_dir == "sessions" else Path(config.session_log_dir) / "clips"
        self.clip_exporter = ClipExporter(clips)
        self._delivery_clip_exporter = DeliveryClipExporter()
        self._match_recorder: MatchRecorder | None = None
        self._pitch_homography: np.ndarray | None = None
        self._drs_call_pending: float | None = None
        self._drs_delivery_id = 0

        hsv = config.ball_hsv or {
            "hmin": 10, "smin": 44, "vmin": 192,
            "hmax": 125, "smax": 114, "vmax": 255,
        }
        self.detector = HybridDetector(
            detection_mode=config.detection_mode,
            hsv_values=hsv,
            rgb_lower=np.array(config.batsman_rgb_lower),
            rgb_upper=np.array(config.batsman_rgb_upper),
            canny1=config.batsman_canny1,
            canny2=config.batsman_canny2,
            yolo_model=config.yolo_model,
            yolo_ball_conf=config.yolo_ball_confidence,
            yolo_person_conf=config.yolo_person_confidence,
            detection_scale=config.detection_scale,
            pitch_cache_frames=config.pitch_cache_frames,
        )

        self._source: VideoSource | None = None
        self._grabber: ThreadedFrameGrabber | None = None
        self._video_fps = 30.0
        self._sequential_play = False
        self._stump_points: StumpPoints | None = None
        self._stump_points_locked = False
        self._stump_from_calibration = False
        self._last_verdict = ""
        self._last_analysis_frame = -1
        self._prime_stumps_on_open = False
        self._last_drs_call_record_path: Path | None = None
        self._load_calibration()

    @property
    def is_live(self) -> bool:
        return self.config.mode == "live"

    @property
    def match_recording_path(self) -> Path | None:
        if self._match_recorder:
            return self._match_recorder.current_path
        return None

    @property
    def recording_elapsed(self) -> float:
        if self._match_recorder:
            return self._match_recorder.elapsed_seconds
        return 0.0

    @property
    def stump_points(self) -> StumpPoints | None:
        return self._stump_points

    @property
    def video_fps(self) -> float:
        return self._video_fps

    @property
    def total_frames(self) -> int:
        if self._source and self.config.mode != "live":
            return self._source.total_frames
        return 0

    def set_stump_points(self, points: StumpPoints | None, *, from_calibration: bool = False) -> None:
        self._stump_points = points
        if points is not None:
            self._stump_points_locked = True
            self._stump_from_calibration = from_calibration

    def reset_auto_stump_lock(self) -> None:
        """Clear auto-derived stumps when opening a new video (keep file calibration)."""
        if self._stump_from_calibration:
            return
        self._stump_points = None
        self._stump_points_locked = False

    def _try_lock_stumps_from_pitch(
        self,
        pitch_contours: list,
        frame_h: int,
        frame_w: int | None = None,
    ) -> None:
        if self._stump_points_locked or self._stump_from_calibration:
            return
        from drs.ui.stumps import stump_points_from_pitch_contour

        derived = stump_points_from_pitch_contour(
            pitch_contours, self.config, frame_h, frame_w,
        )
        if derived is not None:
            self._stump_points = derived
            self._stump_points_locked = True
            logger.info("Locked stump corridor from pitch detection")

    def _load_calibration(self) -> None:
        if not self.config.calibration_file:
            user_cal = user_calibration_dir() / f"{self.config.ground_id}.json"
            if user_cal.is_file():
                self.config.calibration_file = str(user_cal)
            else:
                logger.warning("No stump calibration — using pitch heuristic lines")
                return

        cal_path = Path(self.config.calibration_file)
        if not cal_path.is_absolute():
            cal_path = app_root() / cal_path
        if not cal_path.is_file():
            cal_path = user_calibration_dir() / cal_path.name
        cal = PitchCalibration.load(cal_path)
        if cal is None:
            logger.error("Could not load calibration: %s", cal_path)
            return
        self._stump_points = cal.get_stump_points("primary")
        cam = cal.cameras.get("primary") or next(iter(cal.cameras.values()), None)
        if cam is not None:
            self._pitch_homography = cam.homography
        if self._stump_points:
            self._stump_points_locked = True
            self._stump_from_calibration = True
            logger.info("Loaded stump calibration for ground '%s'", cal.ground_id)

    def open(self) -> tuple[bool, str]:
        if self.config.mode == "live":
            cam = self.config.cameras.get("primary")
            if not cam or not cam.enabled:
                return False, "No camera configured for live mode."
            self._source = VideoSource(cam.type, cam.source, cam.name)
            self._grabber = ThreadedFrameGrabber(self._source)
            if not self._grabber.start():
                return False, "Cannot open live camera."
            self._video_fps = 30.0
            self.reset_delivery()
            self.reset_auto_stump_lock()
            if self.config.recording_enabled:
                out_dir = (
                    Path(self.config.recording_output_dir)
                    if self.config.recording_output_dir
                    else user_matches_dir()
                )
                self._match_recorder = MatchRecorder(
                    self.config.ground_id,
                    self._video_fps,
                    enabled=True,
                    output_dir=out_dir,
                    segment_minutes=self.config.recording_segment_minutes,
                    record_width=self.config.recording_width,
                )
                self._match_recorder.start()
            return True, ""

        path = self.config.video_path
        if not Path(path).is_file():
            return False, f"Video file not found: {path}"
        self._source = VideoSource("file", path, "video")
        if not self._source.open():
            return False, f"Cannot open video: {path}"
        if not self._source.probe():
            self._source.release()
            return False, f"Cannot decode video frames: {path}"
        self._video_fps = max(self._source.fps, 1.0)
        self._sequential_play = False
        self.reset_delivery()
        self.reset_auto_stump_lock()
        self._prime_stumps_on_open = not self._stump_from_calibration and not self._stump_points_locked
        return True, ""

    def _prime_stump_lock_from_video_start(self) -> None:
        """Lock stump corridor from the clearest early frame (wide striker-end pitch row)."""
        if self._stump_from_calibration or self._stump_points_locked:
            return
        if self._source is None or self.config.mode == "live":
            return

        from drs.ui.stumps import score_pitch_frame_for_stump_lock

        best_score = -1.0
        best_contours: list | None = None
        best_shape: tuple[int, int] | None = None
        scan_frames = min(20, max(1, self._source.total_frames))

        for frame_idx in range(scan_frames):
            self._source.seek(frame_idx)
            ret, frame = self._source.read()
            if not ret or frame is None:
                continue
            result = self.detector.process(frame)
            score = score_pitch_frame_for_stump_lock(
                result.pitch_contours,
                self.config,
                frame.shape[0],
                frame.shape[1],
            )
            if score > best_score:
                best_score = score
                best_contours = result.pitch_contours
                best_shape = (frame.shape[0], frame.shape[1])

        if best_contours is not None and best_shape is not None:
            self._try_lock_stumps_from_pitch(
                best_contours, best_shape[0], best_shape[1],
            )

        self._source.seek(0)

    def prime_stumps_if_needed(self) -> None:
        """Scan early frames for stump corridor once (deferred from open)."""
        if not self._prime_stumps_on_open or self.config.mode == "live":
            return
        self._prime_stump_lock_from_video_start()
        self._prime_stumps_on_open = False
        if self._source is not None:
            self._source.seek(0)
            self._sequential_play = False

    def close(self) -> None:
        if self._match_recorder:
            self._match_recorder.stop()
            self._match_recorder = None
        if self._grabber:
            self._grabber.stop()
            self._grabber = None
        if self._source:
            self._source.release()
            self._source = None
        self.session_log.save(user_sessions_dir())

    def reset_delivery(self) -> None:
        self.state.reset_delivery()
        self.event_detector.reset()
        self.detector.reset_tracker()
        self._last_verdict = ""

    def notify_frame_position(self, frame_pos: int) -> None:
        """Reset delivery state when jumping backward or skipping frames."""
        prev = self._last_analysis_frame
        if prev >= 0 and (frame_pos < prev or frame_pos - prev > 1):
            self.reset_delivery()
        self._last_analysis_frame = frame_pos

    def _collect_trajectory(self, x: int, y: int, frame_w: int, frame_h: int) -> None:
        if x == 0 and y == 0:
            return
        if self.state.impact_locked and len(self.state.trajectory_pitch_points) >= 4:
            return
        if not self.event_detector._delivery_in_progress(self.state):
            return

        nx, ny = pixel_to_pitch_normalized(
            x, y,
            frame_w=frame_w,
            frame_h=frame_h,
            homography=self._pitch_homography,
            stump_points=self._stump_points,
        )

        if self.state.trajectory_pitch_points:
            last_nx, last_ny = self.state.trajectory_pitch_points[-1]
            if ny > last_ny + 0.04:
                return
            if abs(nx - last_nx) > 0.14 and abs(ny - last_ny) < 0.025:
                return
            if (nx - last_nx) ** 2 + (ny - last_ny) ** 2 < 0.00025:
                return

        self.state.trajectory_points.append((float(x), float(y)))
        self.state.trajectory_pitch_points.append((nx, ny))

    def trigger_drs_call(self) -> int:
        """Mark a DRS call; clip is finalized after post-roll seconds."""
        self._drs_delivery_id += 1
        self._drs_call_pending = time.time()
        self.state.delivery_count = self._drs_delivery_id
        return self._drs_delivery_id

    def finalize_drs_call_if_ready(self) -> Path | None:
        if self._drs_call_pending is None:
            return None
        if time.time() - self._drs_call_pending < self.config.clip_post_roll_seconds:
            return None
        call_time = self._drs_call_pending
        delivery_id = self._drs_delivery_id
        clip_path = export_drs_call_clip(
            self.ring_buffer.as_list(),
            call_time,
            delivery_id,
            self._video_fps,
            pre_seconds=self.config.clip_pre_roll_seconds,
            post_seconds=self.config.clip_post_roll_seconds,
            exporter=self._delivery_clip_exporter,
        )
        match_path = str(self.match_recording_path) if self.match_recording_path else None
        if clip_path:
            record = DRSCallRecord(
                delivery_id=delivery_id,
                timestamp=call_time,
                clip_path=str(clip_path),
                match_recording_path=match_path,
                trajectory_points=[[p[0], p[1]] for p in self.state.trajectory_pitch_points],
                pitch_point=list(self.state.pitch_point) if self.state.pitch_point else None,
                impact_point=list(self.state.impact_point) if self.state.impact_point else None,
                verdict=self.state.verdict,
                verdict_reason=self.state.verdict_reason,
                confidence_overall=self.state.confidence_overall,
                ai_advisory=None,
            )
            self._last_drs_call_record_path = self._delivery_clip_exporter.save_call_record(record)
        self._drs_call_pending = None
        return clip_path

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.state.update_ball_prev()
        result = self.detector.process(frame)
        self._try_lock_stumps_from_pitch(result.pitch_contours, frame.shape[0], frame.shape[1])
        self.state.ball.x = result.ball_x
        self.state.ball.y = result.ball_y
        self.state.ball.source = result.ball_source
        self.state.ball.confidence = result.ball_confidence
        self.state.update_ball_diff()
        self.event_detector.process_frame(self.state, result.pitch_contours, result.batsman_contours)
        self._collect_trajectory(result.ball_x, result.ball_y, frame.shape[1], frame.shape[0])
        self.verdict_engine.evaluate(self.state, self._stump_points)
        self._maybe_log_delivery()
        display = render_frame(frame, result, self.state, self.config, self._stump_points)
        if self._match_recorder and self.is_live:
            self._match_recorder.write(frame)
        return display

    def _maybe_log_delivery(self) -> None:
        if not self.state.verdict or self.state.verdict == self._last_verdict:
            return
        self._last_verdict = self.state.verdict
        self.session_log.add_delivery(
            DeliveryRecord(
                delivery_id=self.state.delivery_count or len(self.session_log.deliveries) + 1,
                timestamp=time.time(),
                lbw_detected=self.state.lbw_detected,
                pad_detected=self.state.pad_detected,
                pitch_point=self.state.pitch_point,
                impact_point=self.state.impact_point,
                motion_class=self.state.last_motion_class,
                fused_ball=self.state.fused_ball_pitch,
                verdict=self.state.verdict,
                verdict_reason=self.state.verdict_reason,
                confidence_overall=self.state.confidence_overall,
                ai_advisory=self._ai_advisory_dict_from_state(),
            )
        )
        logger.info(
            "Verdict: %s (%.0f%% confidence) — %s",
            self.state.verdict,
            self.state.confidence_overall * 100,
            self.state.verdict_reason,
        )

    def apply_advisory_result(self, result) -> None:
        """Apply AI advisory to runtime state (main thread only)."""
        from drs.services.advisory.models import AdvisoryResult

        if not isinstance(result, AdvisoryResult):
            return
        self.state.ai_verdict = result.recommended_verdict
        self.state.ai_summary = result.summary
        self.state.ai_pending = False

        if (
            self.config.ai_resolve_review
            and self.state.verdict == "REVIEW"
            and result.valid
            and result.recommended_verdict in ("OUT", "NOT OUT")
            and result.confidence >= self.config.ai_min_confidence_auto
        ):
            self.state.verdict_reason = (
                f"AI resolved REVIEW → {result.recommended_verdict}: {result.summary}"
            )

        if self._last_drs_call_record_path and self._last_drs_call_record_path.is_file():
            from drs.recording.delivery_clip import patch_call_record_ai
            patch_call_record_ai(self._last_drs_call_record_path, result.to_dict())

        self._log_ai_audit(result)

    def _log_ai_audit(self, result) -> None:
        from drs.services.advisory.prompts import PROMPT_VERSION

        if not self.session_log.deliveries:
            return
        record = self.session_log.deliveries[-1]
        record.ai_advisory = {
            **result.to_dict(),
            "prompt_version": PROMPT_VERSION,
            "model": self.config.ollama_model,
            "provider": self.config.ai_provider,
        }

    def _ai_advisory_dict_from_state(self) -> dict | None:
        if not self.state.ai_verdict:
            return None
        return {
            "recommended_verdict": self.state.ai_verdict,
            "summary": self.state.ai_summary,
        }

    def read_frame(self, playback: PlaybackState) -> tuple[np.ndarray | None, bool]:
        if playback.mode == ViewMode.BUFFER:
            entry = self.ring_buffer.get_at(playback.buffer_index)
            if entry is None:
                return None, False
            frame = entry.combined_frame if entry.combined_frame is not None else entry.primary_frame
            return frame, True

        if self.config.mode == "live":
            pkt = self._grabber.read_latest() if self._grabber else None
            if pkt is None:
                return None, False
            return pkt.frame, True

        if self._source is None:
            return None, False

        if playback.mode == ViewMode.PLAYING:
            if not self._sequential_play:
                if playback.frame_pos > 0:
                    self._source.seek(playback.frame_pos)
                self._sequential_play = True
            ret, frame = self._source.read()
            return frame, bool(ret and frame is not None)

        self._sequential_play = False
        self._source.seek(playback.frame_pos)
        ret, frame = self._source.read()
        return frame, bool(ret and frame is not None)

    def on_playback_action(self, action: str | None, playback: PlaybackState, prev_mode: ViewMode) -> None:
        if action in ("restart", "jump_start"):
            self.reset_delivery()
            self._last_analysis_frame = -1
        if action in ("restart", "jump_start", "step_back", "step_forward", "jump_end"):
            self._sequential_play = False
        if action == "restart" and self._source:
            self._source.seek(0)
            playback.frame_pos = 0
            self.reset_auto_stump_lock()
            self._prime_stump_lock_from_video_start()
        if action == "pause":
            if playback.mode == ViewMode.PAUSED and prev_mode == ViewMode.PLAYING:
                playback.frame_pos = max(0, playback.frame_pos - 1)
                self._sequential_play = False
            elif playback.mode == ViewMode.PLAYING and prev_mode == ViewMode.PAUSED:
                self._sequential_play = True

    def push_buffer(self, frame: np.ndarray, display: np.ndarray) -> None:
        self.ring_buffer.push(BufferedFrame(
            timestamp=time.time(),
            primary_frame=frame.copy(),
            secondary_frame=None,
            state_snapshot={"verdict": self.state.verdict},
            combined_frame=display.copy(),
        ))

    def save_clip(self, playback: PlaybackState) -> Path | None:
        frames = self.ring_buffer.as_list()
        if not frames:
            return None
        if playback.mode == ViewMode.BUFFER:
            clip = [frames[playback.buffer_index]]
        else:
            clip = frames[-min(150, len(frames)):]
        return self.clip_exporter.export(clip, self.state.delivery_count or 1, self._video_fps)
