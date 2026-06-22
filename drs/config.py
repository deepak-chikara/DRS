"""Configuration loading for DRS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraConfig:
    name: str
    type: str
    source: str | int
    enabled: bool = True


@dataclass
class DRSConfig:
    mode: str = "file"
    video_path: str = "lbw.mp4"
    session_log_dir: str = "sessions"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    detection_mode: str = "color"
    yolo_model: str = "yolov8n.pt"
    yolo_ball_confidence: float = 0.25
    yolo_person_confidence: float = 0.4
    ball_hsv: dict[str, int] = field(default_factory=dict)
    batsman_rgb_lower: list[int] = field(default_factory=lambda: [112, 0, 181])
    batsman_rgb_upper: list[int] = field(default_factory=lambda: [255, 255, 255])
    batsman_canny1: int = 100
    batsman_canny2: int = 200
    pitch_area_min: int = 50000
    batsman_area_min: int = 5000
    pitch_stable_frames: int = 5
    impact_distance_px: int = 15
    impact_batleg_px: int = 25
    stump_width_ratio: float = 0.05
    detection_scale: float = 0.5
    pitch_cache_frames: int = 30
    ring_buffer_seconds: int = 30
    fps_assumed: int = 30
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ring_buffer_capacity(self) -> int:
        return self.ring_buffer_seconds * self.fps_assumed


def load_config(path: str | Path) -> DRSConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    base_dir = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cameras: dict[str, CameraConfig] = {}
    for key, cam in (data.get("cameras") or {}).items():
        source = cam.get("source", "")
        if cam.get("type") == "usb":
            source = int(source)
        cameras[key] = CameraConfig(
            name=cam.get("name", key),
            type=cam.get("type", "file"),
            source=source,
            enabled=cam.get("enabled", True),
        )

    video = data.get("video", "lbw.mp4")
    if not Path(video).is_absolute():
        video = str((base_dir / video).resolve())

    if "primary" not in cameras:
        cameras["primary"] = CameraConfig("primary", "file", video, True)

    detection = data.get("detection", {})
    live = data.get("live", {})
    if data.get("mode") == "live" and live.get("detection_mode"):
        detection_mode = live.get("detection_mode", "hybrid")
    else:
        detection_mode = detection.get("mode", "color")

    return DRSConfig(
        mode=data.get("mode", "file"),
        video_path=video,
        session_log_dir=data.get("session", {}).get("log_dir", "sessions"),
        cameras=cameras,
        detection_mode=detection_mode,
        yolo_model=detection.get("yolo_model", "yolov8n.pt"),
        yolo_ball_confidence=detection.get("yolo_ball_confidence", 0.25),
        yolo_person_confidence=detection.get("yolo_person_confidence", 0.4),
        ball_hsv=data.get("ball", {}).get("hsv", {}),
        batsman_rgb_lower=data.get("batsman", {}).get("rgb_lower", [112, 0, 181]),
        batsman_rgb_upper=data.get("batsman", {}).get("rgb_upper", [255, 255, 255]),
        pitch_area_min=data.get("thresholds", {}).get("pitch_area_min", 50000),
        batsman_area_min=data.get("thresholds", {}).get("batsman_area_min", 5000),
        pitch_stable_frames=data.get("thresholds", {}).get("pitch_stable_frames", 5),
        impact_distance_px=data.get("thresholds", {}).get("impact_distance_px", 15),
        impact_batleg_px=data.get("thresholds", {}).get("impact_batleg_px", 25),
        stump_width_ratio=data.get("thresholds", {}).get("stump_width_ratio", 0.05),
        detection_scale=data.get("performance", {}).get("detection_scale", 0.5),
        pitch_cache_frames=data.get("performance", {}).get("pitch_cache_frames", 30),
        ring_buffer_seconds=data.get("buffer_seconds", 30),
        fps_assumed=30,
        raw=data,
    )
