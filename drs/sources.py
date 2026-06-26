"""Video input sources: file, USB webcam, and RTSP streams."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import cv2


@dataclass
class FramePacket:
    frame: object  # numpy ndarray
    timestamp: float
    frame_index: int
    camera_name: str


class VideoSource:
    """Unified OpenCV video capture for file, USB, or RTSP."""

    def __init__(self, source_type: str, source: str | int, camera_name: str = "primary"):
        self.source_type = source_type
        self.source = source
        self.camera_name = camera_name
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0

    def open(self) -> bool:
        if self.source_type == "usb":
            self._cap = cv2.VideoCapture(int(self.source))
        elif self.source_type == "rtsp":
            self._cap = cv2.VideoCapture(str(self.source))
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            path = str(self.source)
            self._cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(path)

        return self._cap is not None and self._cap.isOpened()

    def probe(self) -> bool:
        """Read one frame to verify the source decodes; rewind file sources to start."""
        if self._cap is None:
            return False
        ret, frame = self.read()
        if not ret or frame is None:
            return False
        if self.source_type == "file":
            self.seek(0)
        return True

    def read(self) -> tuple[bool, object | None]:
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if ret:
            self._frame_index += 1
        return ret, frame

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def total_frames(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 30.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def seek(self, frame_pos: int) -> None:
        if self._cap is not None and self.source_type == "file":
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self._frame_index = frame_pos

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def packets(self) -> Iterator[FramePacket]:
        while True:
            ret, frame = self.read()
            if not ret or frame is None:
                break
            yield FramePacket(
                frame=frame,
                timestamp=time.time(),
                frame_index=self._frame_index,
                camera_name=self.camera_name,
            )


class ThreadedFrameGrabber:
    """Background thread that keeps only the latest frame (drops stale frames)."""

    def __init__(self, source: VideoSource):
        self.source = source
        self._queue: queue.Queue[FramePacket | None] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_packet: FramePacket | None = None

    def start(self) -> bool:
        if not self.source.open():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        return True

    def _grab_loop(self) -> None:
        while self._running:
            ret, frame = self.source.read()
            if not ret or frame is None:
                if self.source.source_type == "file":
                    self._queue.put(None)
                    break
                time.sleep(0.01)
                continue

            packet = FramePacket(
                frame=frame,
                timestamp=time.time(),
                frame_index=self.source.frame_index,
                camera_name=self.source.camera_name,
            )
            try:
                self._queue.put_nowait(packet)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(packet)

    def read_latest(self, timeout: float = 1.0) -> FramePacket | None:
        try:
            packet = self._queue.get(timeout=timeout)
            if packet is None:
                return None
            self._last_packet = packet
            while True:
                try:
                    newer = self._queue.get_nowait()
                    if newer is None:
                        return None
                    self._last_packet = newer
                except queue.Empty:
                    break
            return self._last_packet
        except queue.Empty:
            return self._last_packet

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.source.release()
