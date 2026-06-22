"""Frame synchronization and ring buffer for dual-camera ingest."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from drs.sources import FramePacket


@dataclass
class SyncedFrames:
    primary: FramePacket
    secondary: FramePacket | None
    sync_delta_ms: float | None
    synced: bool


class FrameSynchronizer:
    """Pair frames from two cameras by timestamp or frame index."""

    def __init__(self, max_diff_ms: int = 33, use_frame_index: bool = False):
        self.max_diff_ms = max_diff_ms
        self.use_frame_index = use_frame_index
        self._secondary_buffer: deque[FramePacket] = deque(maxlen=30)

    def add_secondary(self, packet: FramePacket) -> None:
        self._secondary_buffer.append(packet)

    def sync(self, primary: FramePacket, secondary: FramePacket | None) -> SyncedFrames:
        if secondary is None or not self._secondary_buffer:
            return SyncedFrames(
                primary=primary,
                secondary=None,
                sync_delta_ms=None,
                synced=secondary is None,
            )

        best: FramePacket | None = None
        best_delta = float("inf")

        for candidate in self._secondary_buffer:
            if self.use_frame_index:
                delta = abs(candidate.frame_index - primary.frame_index)
                if delta < best_delta:
                    best_delta = delta
                    best = candidate
            else:
                delta_ms = abs((candidate.timestamp - primary.timestamp) * 1000)
                if delta_ms < best_delta:
                    best_delta = delta_ms
                    best = candidate

        if best is None:
            return SyncedFrames(primary=primary, secondary=None, sync_delta_ms=None, synced=False)

        if self.use_frame_index:
            synced = best_delta <= 1
            sync_ms = best_delta * (1000 / 30)
        else:
            synced = best_delta <= self.max_diff_ms
            sync_ms = best_delta

        return SyncedFrames(
            primary=primary,
            secondary=best if synced else None,
            sync_delta_ms=sync_ms,
            synced=synced,
        )


@dataclass
class BufferedFrame:
    timestamp: float
    primary_frame: object
    secondary_frame: object | None
    state_snapshot: dict
    combined_frame: object | None = None


class RingBuffer:
    """Ring buffer storing last N seconds of frames and state for delayed review."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer: deque[BufferedFrame] = deque(maxlen=capacity)

    def push(self, entry: BufferedFrame) -> None:
        self._buffer.append(entry)

    def __len__(self) -> int:
        return len(self._buffer)

    def get_delayed(self, delay_seconds: float) -> list[BufferedFrame]:
        if not self._buffer:
            return []
        cutoff = time.time() - delay_seconds
        return [e for e in self._buffer if e.timestamp <= cutoff]

    def get_recent(self, count: int) -> list[BufferedFrame]:
        return list(self._buffer)[-count:]

    def latest(self) -> BufferedFrame | None:
        return self._buffer[-1] if self._buffer else None

    def as_list(self) -> list[BufferedFrame]:
        return list(self._buffer)

    def get_at(self, index: int) -> BufferedFrame | None:
        if not self._buffer or index < 0 or index >= len(self._buffer):
            return None
        return self._buffer[index]
