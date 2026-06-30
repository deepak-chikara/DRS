"""Per-ground detection and calibration presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from drs.config import DRSConfig, save_config
from drs.paths import user_calibration_dir, user_config_path, user_data_dir


def grounds_dir() -> Path:
    path = user_data_dir() / "grounds"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ground_preset_path(ground_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in ground_id) or "default"
    return grounds_dir() / f"{safe}.json"


def list_ground_ids() -> list[str]:
    ids = sorted(p.stem for p in grounds_dir().glob("*.json"))
    cal_ids = sorted(p.stem for p in user_calibration_dir().glob("*.json"))
    merged = sorted(set(ids) | set(cal_ids))
    return merged or ["default"]


def save_ground_preset(config: DRSConfig) -> Path:
    """Persist HSV, detection, and calibration path for a ground."""
    path = ground_preset_path(config.ground_id)
    payload: dict[str, Any] = {
        "ground_id": config.ground_id,
        "calibration_file": config.calibration_file,
        "detection_mode": config.detection_mode,
        "detection_scale": config.detection_scale,
        "ball_hsv": dict(config.ball_hsv),
        "yolo_model": config.yolo_model,
        "yolo_ball_confidence": config.yolo_ball_confidence,
        "yolo_person_confidence": config.yolo_person_confidence,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_ground_preset(ground_id: str) -> dict[str, Any] | None:
    path = ground_preset_path(ground_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def apply_ground_preset(config: DRSConfig, ground_id: str) -> bool:
    """Apply saved preset; returns True if a preset was found."""
    preset = load_ground_preset(ground_id)
    if preset is None:
        config.ground_id = ground_id
        cal = user_calibration_dir() / f"{ground_id}.json"
        if cal.is_file():
            config.calibration_file = str(cal)
        return False

    config.ground_id = ground_id
    if preset.get("calibration_file"):
        config.calibration_file = preset["calibration_file"]
    else:
        cal = user_calibration_dir() / f"{ground_id}.json"
        if cal.is_file():
            config.calibration_file = str(cal)
    config.detection_mode = preset.get("detection_mode", config.detection_mode)
    config.detection_scale = float(preset.get("detection_scale", config.detection_scale))
    if preset.get("ball_hsv"):
        config.ball_hsv = dict(preset["ball_hsv"])
    config.yolo_model = preset.get("yolo_model", config.yolo_model)
    config.yolo_ball_confidence = float(
        preset.get("yolo_ball_confidence", config.yolo_ball_confidence)
    )
    config.yolo_person_confidence = float(
        preset.get("yolo_person_confidence", config.yolo_person_confidence)
    )
    return True


def switch_ground(config: DRSConfig, ground_id: str) -> None:
    apply_ground_preset(config, ground_id)
    save_ground_preset(config)
    save_config(config, user_config_path())
