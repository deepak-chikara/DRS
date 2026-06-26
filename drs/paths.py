"""Application paths for install dir and per-user data."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def app_root() -> Path:
    """Project or PyInstaller bundle root."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """Per-user writable data: %LOCALAPPDATA%\\DRS on Windows."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    path = Path(base) / "DRS"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_config_path() -> Path:
    return user_data_dir() / "settings.yaml"


def user_logs_dir() -> Path:
    path = user_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_clips_dir() -> Path:
    path = user_data_dir() / "clips"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_matches_dir() -> Path:
    path = user_data_dir() / "matches"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_sessions_dir() -> Path:
    path = user_data_dir() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_calibration_dir() -> Path:
    path = user_data_dir() / "calibration"
    path.mkdir(parents=True, exist_ok=True)
    return path


def bundled_config_path(name: str = "default.yaml") -> Path:
    return app_root() / "config" / name


def ensure_user_config() -> Path:
    """Copy bundled default.yaml to user profile on first run."""
    dest = user_config_path()
    if dest.is_file():
        return dest
    src = bundled_config_path("default.yaml")
    if src.is_file():
        shutil.copy2(src, dest)
        data = dest.read_text(encoding="utf-8")
        data = data.replace('calibration_file: ""', f'calibration_file: ""')
        dest.write_text(data, encoding="utf-8")
    return dest
