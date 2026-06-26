"""Continuous MP4 recording for live match sessions."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from drs.paths import user_matches_dir


class MatchRecorder:
    """Write live camera frames to segmented MP4 files on disk."""

    def __init__(
        self,
        ground_id: str,
        fps: float = 30.0,
        *,
        enabled: bool = True,
        output_dir: Path | None = None,
        segment_minutes: int = 45,
        record_width: int | None = None,
    ):
        self.ground_id = ground_id or "match"
        self.fps = max(1.0, fps)
        self.enabled = enabled
        self.output_dir = output_dir or user_matches_dir()
        self.segment_minutes = max(1, segment_minutes)
        self.record_width = record_width

        self._writer: cv2.VideoWriter | None = None
        self._segment_start = 0.0
        self._session_start = 0.0
        self._frame_size: tuple[int, int] | None = None
        self._segment_index = 0
        self._current_path: Path | None = None
        self._frames_written = 0

    @property
    def current_path(self) -> Path | None:
        return self._current_path

    @property
    def elapsed_seconds(self) -> float:
        if self._session_start <= 0:
            return 0.0
        return time.time() - self._session_start

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def start(self) -> None:
        if not self.enabled:
            return
        self._session_start = time.time()
        self._segment_start = self._session_start
        self._segment_index = 0
        self._frames_written = 0

    def _open_segment(self) -> Path | None:
        self._close_writer()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_seg{self._segment_index}" if self._segment_index else ""
        path = self.output_dir / f"{self.ground_id}_{stamp}{suffix}.mp4"
        if self._frame_size is None:
            self._current_path = path
            return path
        self._writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self._frame_size,
        )
        self._current_path = path
        self._segment_start = time.time()
        return path

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        if self.record_width and frame.shape[1] > self.record_width:
            scale = self.record_width / frame.shape[1]
            h = max(1, int(frame.shape[0] * scale))
            out = cv2.resize(frame, (self.record_width, h), interpolation=cv2.INTER_AREA)
        if self._frame_size is None:
            h, w = out.shape[:2]
            self._frame_size = (w, h)
            self._open_segment()
        elif (out.shape[1], out.shape[0]) != self._frame_size:
            out = cv2.resize(out, self._frame_size, interpolation=cv2.INTER_AREA)
        return out

    def write(self, frame: np.ndarray) -> None:
        if not self.enabled or frame is None:
            return
        prepared = self._prepare_frame(frame)
        if self._writer is None:
            return
        self._writer.write(prepared)
        self._frames_written += 1
        if time.time() - self._segment_start >= self.segment_minutes * 60:
            self._segment_index += 1
            self._open_segment()

    def stop(self) -> Path | None:
        path = self._current_path
        self._close_writer()
        self._session_start = 0.0
        return path
