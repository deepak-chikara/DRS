"""Session logging and clip export."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2


@dataclass
class DeliveryRecord:
    delivery_id: int
    timestamp: float
    lbw_detected: bool
    pad_detected: bool
    pitch_point: tuple[int, int] | None
    impact_point: tuple[int, int] | None
    motion_class: str
    fused_ball: tuple[float, float] | None = None
    verdict: str = ""
    verdict_reason: str = ""
    confidence_overall: float = 0.0
    ai_advisory: dict | None = None


@dataclass
class SessionLog:
    ground_name: str
    started_at: float = field(default_factory=time.time)
    deliveries: list[DeliveryRecord] = field(default_factory=list)

    def add_delivery(self, record: DeliveryRecord) -> None:
        self.deliveries.append(record)

    def save(self, log_dir: str | Path) -> Path:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        filename = log_path / f"session_{int(self.started_at)}.json"
        data = {
            "ground_name": self.ground_name,
            "started_at": self.started_at,
            "deliveries": [asdict(d) for d in self.deliveries],
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filename


class ClipExporter:
    """Export MP4 snippet from buffered frames."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, frames: list, delivery_id: int, fps: float = 30.0) -> Path | None:
        if not frames:
            return None

        sample = frames[0].primary_frame
        h, w = sample.shape[:2]
        out_path = self.output_dir / f"delivery_{delivery_id}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for entry in frames:
            writer.write(entry.primary_frame)
        writer.release()
        return out_path
