# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for DRS Pro. Run: pyinstaller build/drs.spec"""

import sys
from pathlib import Path

block_cipher = None
root = Path(SPECPATH).parent.parent

a = Analysis(
    [str(root / "drs" / "app" / "main.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[
        (str(root / "config"), "config"),
        (str(root / "drs" / "app" / "theme.qss"), "drs/app"),
        (str(root / "docs" / "USER_GUIDE.md"), "docs"),
    ],
    hiddenimports=[
        "cv2", "numpy", "cvzone", "yaml", "PySide6", "PySide6.QtCore",
        "PySide6.QtGui", "PySide6.QtWidgets", "ultralytics",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name="DRS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DRS",
)
