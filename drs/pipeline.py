"""DRS video player with LBW analysis on pause."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drs.config import DRSConfig, load_config
from drs.decision.events import EventDetector
from drs.detectors.hybrid import HybridDetector
from drs.session.logger import ClipExporter, SessionLog
from drs.sources import ThreadedFrameGrabber, VideoSource
from drs.state import DRSState
from drs.sync.frame_sync import BufferedFrame, RingBuffer
from drs.ui.display import attach_controls_sidebar, draw_overlays, draw_status_banner
from drs.ui.playback import PlaybackState, ViewMode, decode_key, handle_playback_key


class DRSPipeline:
    def __init__(self, config: DRSConfig):
        self.config = config
        self.state = DRSState()
        self.event_detector = EventDetector(config)
        self.ring_buffer = RingBuffer(config.ring_buffer_capacity)
        self.session_log = SessionLog(ground_name="DRS Session")
        self.clip_exporter = ClipExporter(Path(config.session_log_dir) / "clips")

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
        self._buffer_tick = 0
        self._sequential_play = False
        self._last_display = None

    def _setup(self) -> bool:
        if self.config.mode == "live":
            cam = self.config.cameras.get("primary")
            if not cam or not cam.enabled:
                print("No camera configured for live mode.")
                return False
            self._source = VideoSource(cam.type, cam.source, cam.name)
            self._grabber = ThreadedFrameGrabber(self._source)
            return self._grabber.start()

        cam = self.config.cameras.get("primary")
        path = cam.source if cam else self.config.video_path
        self._source = VideoSource("file", path, "video")
        if not self._source.open():
            print(f"Cannot open video: {path}")
            return False
        self._video_fps = max(self._source.fps, 1.0)
        return True

    def _read_frame(self, playback: PlaybackState):
        if self.config.mode == "live":
            pkt = self._grabber.read_latest() if self._grabber else None
            if pkt is None:
                return None, False
            return pkt.frame, True

        if self._source is None:
            return None, False

        # Sequential read while playing (fast). Seek only when paused/stepping.
        if playback.mode == ViewMode.PLAYING:
            if not self._sequential_play:
                self._source.seek(playback.frame_pos)
                self._sequential_play = True
            ret, frame = self._source.read()
            return frame, ret and frame is not None

        self._sequential_play = False
        self._source.seek(playback.frame_pos)
        ret, frame = self._source.read()
        return frame, ret and frame is not None

    def _on_playback_action(self, action: str | None, playback: PlaybackState, prev_mode: ViewMode) -> None:
        if action in ("restart", "jump_start", "step_back", "step_forward", "jump_end"):
            self._sequential_play = False
        if action == "restart" and self._source:
            self._source.seek(0)
            playback.frame_pos = 0
        if action == "pause":
            if playback.mode == ViewMode.PAUSED and prev_mode == ViewMode.PLAYING:
                playback.frame_pos = max(0, playback.frame_pos - 1)
                self._sequential_play = False
            elif playback.mode == ViewMode.PLAYING and prev_mode == ViewMode.PAUSED:
                self._sequential_play = True

    def _analyze(self, frame) -> np.ndarray:
        self.state.update_ball_prev()
        result = self.detector.process(frame)
        self.state.ball.x = result.ball_x
        self.state.ball.y = result.ball_y
        self.state.ball.source = result.ball_source
        self.state.update_ball_diff()
        self.event_detector.process_frame(self.state, result.pitch_contours, result.batsman_contours)
        return draw_overlays(frame, result, self.state, self.config)

    def _wait_ms(self, playback: PlaybackState) -> int:
        if playback.mode != ViewMode.PLAYING:
            return 30
        if self.config.mode == "file" and self._video_fps > 0:
            return max(1, int(1000 / self._video_fps))
        return 1

    def _mode_label(self, playback: PlaybackState) -> str:
        if self.config.mode == "live":
            return "live" if playback.mode == ViewMode.PLAYING else playback.mode.value
        return playback.mode.value

    def _save_clip(self, playback: PlaybackState) -> None:
        frames = self.ring_buffer.as_list()
        if not frames:
            print("Nothing in buffer to save.")
            return
        if playback.mode == ViewMode.BUFFER:
            clip = [frames[playback.buffer_index]]
        else:
            clip = frames[-min(150, len(frames)):]
        path = self.clip_exporter.export(clip, 1, self._video_fps)
        if path:
            print(f"Saved: {path}")

    def run(self) -> None:
        if not self._setup():
            return

        playback = PlaybackState()
        is_live = self.config.mode == "live"
        total = self._source.total_frames if self._source and not is_live else 0
        window = "DRS"

        print("DRS ready. Space=pause to analyse a delivery. Controls shown on screen.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            buffer_len = len(self.ring_buffer)
            frame_info = ""
            display = None

            if playback.mode == ViewMode.BUFFER and buffer_len > 0:
                playback.buffer_index = min(playback.buffer_index, buffer_len - 1)
                entry = self.ring_buffer.get_at(playback.buffer_index)
                if entry:
                    display = entry.combined_frame.copy() if entry.combined_frame is not None else entry.primary_frame
                    frame_info = f"buffer {playback.buffer_index + 1}/{buffer_len}"
            else:
                frame, ok = self._read_frame(playback)
                if not ok:
                    if is_live:
                        time.sleep(0.01)
                        continue
                    if playback.mode == ViewMode.PLAYING:
                        playback.mode = ViewMode.PAUSED
                        self._sequential_play = False
                        if self._last_display is not None:
                            display = self._last_display
                            frame_info = "end of video — press R to restart"
                        else:
                            continue
                    else:
                        break
                else:
                    if playback.mode == ViewMode.PLAYING and not is_live:
                        display = frame
                        frame_info = f"frame {playback.frame_pos + 1}/{total}"
                        playback.frame_pos += 1
                    else:
                        display = self._analyze(frame)
                        frame_info = f"frame {playback.frame_pos + 1}/{total}" if not is_live else "live"

                    self._buffer_tick += 1
                    if is_live or playback.is_analyzing():
                        self.ring_buffer.push(BufferedFrame(
                            timestamp=time.time(),
                            primary_frame=frame.copy(),
                            secondary_frame=None,
                            state_snapshot={},
                            combined_frame=display.copy() if playback.is_analyzing() else None,
                        ))

            if display is None:
                continue

            self._last_display = display

            out = attach_controls_sidebar(draw_status_banner(
                display, self._mode_label(playback), frame_info=frame_info,
            ))
            cv2.imshow(window, out)

            prev_mode = playback.mode
            action = handle_playback_key(
                decode_key(cv2.waitKey(self._wait_ms(playback))),
                playback, is_live=is_live, buffer_len=buffer_len, total_frames=total,
            )
            self._on_playback_action(action, playback, prev_mode)
            if action == "quit":
                break
            if action == "save_clip":
                self._save_clip(playback)

        if self._grabber:
            self._grabber.stop()
        if self._source:
            self._source.release()
        cv2.destroyAllWindows()

        if self.state.lbw_detected:
            print("Potential LBW detected during this session.")
        else:
            print("Session ended.")


def run_pipeline(config_path: str) -> None:
    DRSPipeline(load_config(config_path)).run()
