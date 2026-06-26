"""Export delivery clips from the live ring buffer around a DRS call."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from drs.paths import user_clips_dir, user_sessions_dir
from drs.sync.frame_sync import BufferedFrame


@dataclass
class DRSCallRecord:
    delivery_id: int
    timestamp: float
    clip_path: str
    match_recording_path: str | None
    trajectory_points: list[list[float]]
    pitch_point: list[int] | None
    impact_point: list[int] | None
    verdict: str
    verdict_reason: str


class DeliveryClipExporter:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or user_clips_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_frames(
        self,
        frames: list[BufferedFrame],
        delivery_id: int,
        fps: float = 30.0,
        *,
        use_overlays: bool = True,
    ) -> Path | None:
        if not frames:
            return None
        sample = frames[0].combined_frame if use_overlays and frames[0].combined_frame is not None else frames[0].primary_frame
        h, w = sample.shape[:2]
        out_path = self.output_dir / f"drs_call_{delivery_id}_{int(time.time())}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for entry in frames:
            frame = entry.combined_frame if use_overlays and entry.combined_frame is not None else entry.primary_frame
            if frame is not None:
                writer.write(frame)
        writer.release()
        return out_path

    def save_call_record(self, record: DRSCallRecord) -> Path:
        session_dir = user_sessions_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"drs_call_{record.delivery_id}_{int(record.timestamp)}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2)
        return path


def slice_buffer_around_time(
    buffer: list[BufferedFrame],
    call_time: float,
    pre_seconds: float,
    post_seconds: float,
) -> list[BufferedFrame]:
    """Return buffer frames within [call_time - pre, call_time + post]."""
    if not buffer:
        return []
    start = call_time - pre_seconds
    end = call_time + post_seconds
    return [f for f in buffer if start <= f.timestamp <= end]


def export_drs_call_clip(
    buffer: list[BufferedFrame],
    call_time: float,
    delivery_id: int,
    fps: float,
    *,
    pre_seconds: float = 12.0,
    post_seconds: float = 8.0,
    exporter: DeliveryClipExporter | None = None,
) -> Path | None:
    frames = slice_buffer_around_time(buffer, call_time, pre_seconds, post_seconds)
    if not frames:
        frames = buffer[-min(len(buffer), int((pre_seconds + post_seconds) * fps)) :]
    exp = exporter or DeliveryClipExporter()
    return exp.export_frames(frames, delivery_id, fps, use_overlays=True)
