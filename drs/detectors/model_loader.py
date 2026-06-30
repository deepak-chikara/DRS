"""Resolve YOLO model path and optional first-run download."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("drs.detectors")


def user_models_dir() -> Path:
    from drs.paths import user_data_dir

    path = user_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def bundled_models_dir() -> Path:
    from drs.paths import app_root

    return app_root() / "models"


def resolve_yolo_model(model_name: str = "yolov8n.pt") -> str:
    """
    Return a filesystem path to the YOLO weights.
    Search order: user models dir, bundled models, cwd, then trigger download to user dir.
    """
    name = Path(model_name).name
    candidates = [
        user_models_dir() / name,
        bundled_models_dir() / name,
        Path(model_name),
        Path.cwd() / name,
    ]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.insert(1, exe_dir / "models" / name)

    for path in candidates:
        if path.is_file():
            return str(path.resolve())

    return ensure_yolo_model(name)


def ensure_yolo_model(model_name: str = "yolov8n.pt") -> str:
    """Download YOLO weights into user profile on first use (requires network once)."""
    dest = user_models_dir() / model_name
    if dest.is_file():
        return str(dest)

    bundled = bundled_models_dir() / model_name
    if bundled.is_file():
        shutil.copy2(bundled, dest)
        return str(dest)

    try:
        from ultralytics import YOLO

        logger.info("Downloading YOLO model %s (first run)", model_name)
        model = YOLO(model_name)
        src = Path(getattr(model, "ckpt_path", model_name))
        if src.is_file() and src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        elif not dest.is_file():
            YOLO(str(dest))  # triggers download to dest if ultralytics supports path
        logger.info("YOLO model ready at %s", dest)
    except Exception as exc:
        logger.warning("YOLO download failed: %s — hybrid mode may fall back to color", exc)
        return model_name

    return str(dest) if dest.is_file() else model_name
