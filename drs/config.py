"""Configuration loading for DRS."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from drs.paths import user_calibration_dir, user_clips_dir, user_config_path, user_matches_dir, user_sessions_dir

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
    delivery_motion_min_px: int = 4
    delivery_motion_frames: int = 2
    impact_distance_px: int = 15
    impact_batleg_px: int = 25
    stump_width_ratio: float = 0.05
    detection_scale: float = 0.5
    pitch_cache_frames: int = 30
    ring_buffer_seconds: int = 30
    fps_assumed: int = 30
    ground_id: str = "default"
    calibration_file: str = ""
    recording_enabled: bool = True
    recording_output_dir: str = ""
    recording_segment_minutes: int = 45
    recording_width: int | None = 1280
    clip_pre_roll_seconds: float = 12.0
    clip_post_roll_seconds: float = 8.0
    diagram_enabled: bool = True
    ai_enabled: bool = False
    ai_provider: str = "ollama"
    ai_min_confidence_auto: float = 0.72
    ai_resolve_review: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout_seconds: float = 25.0
    ollama_temperature: float = 0.1
    ai_live_enabled: bool = True
    ai_live_interval_seconds: float = 2.0
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

    mode = data.get("mode", "file")

    cameras: dict[str, CameraConfig] = {}
    if mode == "live":
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

    if mode == "file" or "primary" not in cameras:
        cameras["primary"] = CameraConfig("primary", "file", video, True)

    ground_id = data.get("ground_id", "default")
    calibration_file = data.get("calibration_file", "")
    if not calibration_file:
        session_cal = (data.get("session") or {}).get("calibration_file", "")
        calibration_file = session_cal
    if calibration_file and not Path(calibration_file).is_absolute():
        calibration_file = str((base_dir / calibration_file).resolve())

    detection = data.get("detection", {})
    live = data.get("live", {})
    if data.get("mode") == "live" and live.get("detection_mode"):
        detection_mode = live.get("detection_mode", "hybrid")
    else:
        detection_mode = detection.get("mode", "color")

    recording = data.get("recording", {})
    rec_dir = recording.get("output_dir", "")
    if rec_dir and not Path(rec_dir).is_absolute():
        rec_dir = str((base_dir / rec_dir).resolve())

    return DRSConfig(
        mode=mode,
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
        delivery_motion_min_px=data.get("thresholds", {}).get("delivery_motion_min_px", 4),
        delivery_motion_frames=data.get("thresholds", {}).get("delivery_motion_frames", 2),
        impact_distance_px=data.get("thresholds", {}).get("impact_distance_px", 15),
        impact_batleg_px=data.get("thresholds", {}).get("impact_batleg_px", 25),
        stump_width_ratio=data.get("thresholds", {}).get("stump_width_ratio", 0.05),
        detection_scale=data.get("performance", {}).get("detection_scale", 0.5),
        pitch_cache_frames=data.get("performance", {}).get("pitch_cache_frames", 30),
        ring_buffer_seconds=data.get("buffer_seconds", 30),
        fps_assumed=30,
        ground_id=ground_id,
        calibration_file=calibration_file,
        recording_enabled=recording.get("enabled", mode == "live"),
        recording_output_dir=rec_dir,
        recording_segment_minutes=recording.get("segment_minutes", 45),
        recording_width=recording.get("width"),
        clip_pre_roll_seconds=float(data.get("clip", {}).get("pre_roll_seconds", 12)),
        clip_post_roll_seconds=float(data.get("clip", {}).get("post_roll_seconds", 8)),
        diagram_enabled=data.get("diagram", {}).get("enabled", True),
        ai_enabled=data.get("ai", {}).get("enabled", False),
        ai_provider=data.get("ai", {}).get("provider", "ollama"),
        ai_min_confidence_auto=float(data.get("ai", {}).get("min_confidence_auto", 0.72)),
        ai_resolve_review=data.get("ai", {}).get("resolve_review", True),
        ollama_base_url=data.get("ai", {}).get("ollama", {}).get("base_url", "http://127.0.0.1:11434"),
        ollama_model=data.get("ai", {}).get("ollama", {}).get("model", "llama3.2"),
        ollama_timeout_seconds=float(data.get("ai", {}).get("ollama", {}).get("timeout_seconds", 25)),
        ollama_temperature=float(data.get("ai", {}).get("ollama", {}).get("temperature", 0.1)),
        ai_live_enabled=data.get("ai", {}).get("live_enabled", True),
        ai_live_interval_seconds=float(data.get("ai", {}).get("live_interval_seconds", 2.0)),
        raw=data,
    )


def save_config(config: DRSConfig, path: str | Path) -> None:
    """Persist DRSConfig fields to YAML."""
    path = Path(path)
    data = dict(config.raw) if config.raw else {}
    data["mode"] = config.mode
    data["video"] = config.video_path
    data["ground_id"] = config.ground_id
    data["calibration_file"] = config.calibration_file
    data.setdefault("detection", {})["mode"] = config.detection_mode
    data.setdefault("ball", {})["hsv"] = config.ball_hsv
    data.setdefault("batsman", {})["rgb_lower"] = config.batsman_rgb_lower
    data["batsman"]["rgb_upper"] = config.batsman_rgb_upper
    data.setdefault("thresholds", {})
    data["thresholds"].update({
        "pitch_area_min": config.pitch_area_min,
        "batsman_area_min": config.batsman_area_min,
        "pitch_stable_frames": config.pitch_stable_frames,
        "delivery_motion_min_px": config.delivery_motion_min_px,
        "delivery_motion_frames": config.delivery_motion_frames,
        "impact_distance_px": config.impact_distance_px,
        "impact_batleg_px": config.impact_batleg_px,
        "stump_width_ratio": config.stump_width_ratio,
    })
    data.setdefault("performance", {})["detection_scale"] = config.detection_scale
    data["performance"]["pitch_cache_frames"] = config.pitch_cache_frames
    data["buffer_seconds"] = config.ring_buffer_seconds
    data.setdefault("recording", {})
    data["recording"].update({
        "enabled": config.recording_enabled,
        "output_dir": config.recording_output_dir,
        "segment_minutes": config.recording_segment_minutes,
        "width": config.recording_width,
    })
    data.setdefault("clip", {})
    data["clip"].update({
        "pre_roll_seconds": config.clip_pre_roll_seconds,
        "post_roll_seconds": config.clip_post_roll_seconds,
    })
    data.setdefault("diagram", {})["enabled"] = config.diagram_enabled
    data.setdefault("ai", {})
    data["ai"].update({
        "enabled": config.ai_enabled,
        "provider": config.ai_provider,
        "min_confidence_auto": config.ai_min_confidence_auto,
        "resolve_review": config.ai_resolve_review,
    })
    data["ai"].setdefault("ollama", {})
    data["ai"]["ollama"].update({
        "base_url": config.ollama_base_url,
        "model": config.ollama_model,
        "timeout_seconds": config.ollama_timeout_seconds,
        "temperature": config.ollama_temperature,
    })
    data["ai"]["live_enabled"] = config.ai_live_enabled
    data["ai"]["live_interval_seconds"] = config.ai_live_interval_seconds
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_user_config() -> DRSConfig:
    """Load config from user profile, falling back to bundled default."""
    from drs.paths import ensure_user_config

    path = ensure_user_config()
    cfg = load_config(path)
    cfg.session_log_dir = str(user_sessions_dir())
    if cfg.calibration_file:
        cal = Path(cfg.calibration_file)
        if not cal.is_file():
            user_cal = user_calibration_dir() / f"{cfg.ground_id}.json"
            if user_cal.is_file():
                cfg.calibration_file = str(user_cal)
    return cfg


def resolve_calibration_path(ground_id: str) -> str:
    return str(user_calibration_dir() / f"{ground_id}.json")
