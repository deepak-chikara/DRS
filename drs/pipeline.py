"""DRS video player with LBW analysis (legacy OpenCV UI)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drs.config import DRSConfig, load_config
from drs.engine import DRSEngine
from drs.logging_setup import setup_logging
from drs.ui.display import attach_controls_sidebar, draw_status_banner
from drs.ui.playback import PlaybackState, ViewMode, decode_key, handle_playback_key


class DRSPipeline:
    """OpenCV window loop — delegates to DRSEngine."""

    def __init__(self, config: DRSConfig):
        setup_logging()
        self.config = config
        self.engine = DRSEngine(config)
        self._last_display = None

    def _frame_label(self, frame_pos: int, total: int) -> str:
        if total > 0:
            return f"frame {frame_pos + 1}/{total}"
        return f"frame {frame_pos + 1}"

    def run(self) -> None:
        ok, err = self.engine.open()
        if not ok:
            print(err)
            return

        playback = PlaybackState()
        is_live = self.config.mode == "live"
        total = self.engine.total_frames
        window = "DRS"

        print("DRS ready. Stump lines, ball/batsman tracking, and OUT/NOT OUT assist active.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.waitKey(1)

        while True:
            buffer_len = len(self.engine.ring_buffer)
            frame_info = ""
            display = None

            if playback.mode == ViewMode.BUFFER and buffer_len > 0:
                playback.buffer_index = min(playback.buffer_index, buffer_len - 1)
                entry = self.engine.ring_buffer.get_at(playback.buffer_index)
                if entry:
                    display = entry.combined_frame.copy() if entry.combined_frame is not None else entry.primary_frame
                    frame_info = f"buffer {playback.buffer_index + 1}/{buffer_len}"
            else:
                frame, read_ok = self.engine.read_frame(playback)
                if not read_ok:
                    if is_live:
                        time.sleep(0.01)
                        continue
                    if playback.mode == ViewMode.PLAYING:
                        if self._last_display is None:
                            print(f"Cannot read video frames: {self.config.video_path}")
                            break
                        playback.mode = ViewMode.PAUSED
                        display = self._last_display
                        frame_info = "end of video — press R to restart"
                    else:
                        print(f"Cannot read frame {playback.frame_pos + 1}/{total or '?'}")
                        break
                else:
                    self.engine.notify_frame_position(playback.frame_pos)
                    display = self.engine.process_frame(frame)
                    if playback.mode == ViewMode.PLAYING and not is_live:
                        frame_info = self._frame_label(playback.frame_pos, total)
                        playback.frame_pos += 1
                    else:
                        frame_info = self._frame_label(playback.frame_pos, total) if not is_live else "live"
                    self.engine.push_buffer(frame, display)

            if display is None:
                continue

            self._last_display = display
            mode = "live" if is_live and playback.mode == ViewMode.PLAYING else playback.mode.value
            out = attach_controls_sidebar(draw_status_banner(display, mode, frame_info=frame_info))
            cv2.imshow(window, out)

            wait_ms = 30 if playback.mode != ViewMode.PLAYING else max(1, int(1000 / self.engine.video_fps))
            prev_mode = playback.mode
            action = handle_playback_key(
                decode_key(cv2.waitKey(wait_ms)),
                playback, is_live=is_live, buffer_len=buffer_len, total_frames=total,
            )
            self.engine.on_playback_action(action, playback, prev_mode)
            if action == "quit":
                break
            if action == "save_clip":
                path = self.engine.save_clip(playback)
                if path:
                    print(f"Saved: {path}")

        self.engine.close()
        cv2.destroyAllWindows()

        if self.engine.state.verdict == "OUT":
            print(f"Session ended — verdict: OUT ({self.engine.state.verdict_reason})")
        else:
            print("Session ended.")


def run_pipeline(config_path: str) -> None:
    DRSPipeline(load_config(config_path)).run()
