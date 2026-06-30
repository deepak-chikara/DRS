# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for DRS Pro — one-folder bundle with all dependencies."""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
root = Path(SPECPATH).parent
icon = root / "assets" / "icon.ico"
version_file = root / "build" / "version_info.txt"

# PySide6 Qt plugins, translations, etc.
pyside6_datas, pyside6_binaries, pyside6_hiddenimports = collect_all("PySide6")

# Optional heavy ML stack — pulled in via ultralytics if installed in build env.
# Omit from collect_all to keep bundle smaller; hybrid mode falls back to color + first-run YOLO download.
optional_collects = []
for pkg in ("cv2", "httpx", "cvzone"):
    try:
        d, b, h = collect_all(pkg)
        optional_collects.append((d, b, h))
    except Exception:
        pass

extra_datas = []
extra_binaries = []
extra_hidden = []
for d, b, h in optional_collects:
    extra_datas.extend(d)
    extra_binaries.extend(b)
    extra_hidden.extend(h)

datas = [
    (str(root / "config"), "config"),
    (str(root / "drs" / "app" / "theme.qss"), "drs/app"),
    (str(root / "docs" / "USER_GUIDE.md"), "docs"),
    (str(root / "docs" / "DEPLOYMENT.md"), "docs"),
    (str(root / "docs" / "EULA.md"), "docs"),
    (str(root / "docs" / "THIRD_PARTY_NOTICES.md"), "docs"),
    (str(root / "assets" / "icon.ico"), "assets"),
    (str(root / "assets" / "icon.png"), "assets"),
] + pyside6_datas + extra_datas

hiddenimports = [
    "yaml",
    "numpy",
    "ball_detect",
    "batsman",
    "pitch",
    "drs",
    "drs.app.branding",
    "drs.detectors.model_loader",
    "drs.detectors.hybrid",
    "drs.detectors.yolo_detector",
    "drs.services.advisory.factory",
    "drs.services.advisory.ollama_service",
    "drs.services.advisory.confidence",
    "drs.services.advisory.evidence",
    "drs.services.advisory.prompts",
    "ultralytics",
] + list(collect_submodules("drs")) + pyside6_hiddenimports + extra_hidden

a = Analysis(
    [str(root / "main.py")],
    pathex=[str(root)],
    binaries=extra_binaries + pyside6_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(root / "build" / "runtime_hook_qt.py")],
    excludes=["matplotlib", "tkinter", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DRS Pro",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon) if icon.is_file() else None,
    version=str(version_file) if version_file.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DRS Pro",
)
