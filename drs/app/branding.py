"""Application icon and display name for dev and PyInstaller bundles."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon

from drs import __app_name__
from drs.paths import app_root, install_dir


def icon_candidates() -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for root in (app_root(), install_dir()):
        for rel in (
            "assets/icon.ico",
            "assets/icon.png",
            "icon.ico",
        ):
            p = (root / rel).resolve()
            if p not in seen and p.is_file():
                seen.add(p)
                out.append(p)
    return out


def load_app_icon() -> QIcon:
    for path in icon_candidates():
        icon = QIcon(str(path))
        if not icon.isNull():
            return icon
    return QIcon()


def apply_app_branding(app) -> QIcon:
    """Set application name and icon (call once after QApplication is created)."""
    icon = load_app_icon()
    app.setApplicationName(__app_name__)
    app.setApplicationDisplayName(__app_name__)
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(  # type: ignore[attr-defined]
                "DRS.Pro.LBWReview.1"
            )
        except Exception:
            pass
    if not icon.isNull():
        app.setWindowIcon(icon)
    return icon
