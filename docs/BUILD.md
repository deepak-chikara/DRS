# DRS Pro — Windows install bundle

## Quick build (one command)

From the `DRS` folder in PowerShell:

```powershell
.\tools\build_installer.ps1
```

This produces:

| Output | Description |
|--------|-------------|
| `dist\DRS Pro\DRS Pro.exe` | Portable app — entire folder is self-contained |
| `dist\installer\DRS-Pro-Setup-1.0.0.exe` | One-click installer (requires [Inno Setup 6](https://jrsoftware.org/isinfo.php)) |

## Install for end users

**Option A — Installer (recommended)**  
Run `DRS-Pro-Setup-1.0.0.exe` → follow wizard → launch **DRS Pro** from Start Menu or desktop.

**Option B — Portable**  
Copy the `dist\DRS Pro` folder anywhere and run `DRS Pro.exe`.

No Python install required. Settings, calibration, and clips are stored in `%LOCALAPPDATA%\DRS\`.

## What's bundled

- DRS Pro Qt desktop app with icon and Windows product name
- OpenCV, NumPy, PySide6, cvzone, httpx
- Ultralytics YOLO (hybrid mode; weights download once to `%LOCALAPPDATA%\DRS\models\` on first use if not cached)
- Default config and user guide

## Requirements to build

- Python 3.10+ with project dependencies (`pip install -r requirements.txt pyinstaller`)
- Optional: Inno Setup 6 for the setup EXE

## Icon / name not showing?

Rebuild after pulling latest changes — the exe embeds `assets/icon.ico` and Windows version info (`ProductName: DRS Pro`). Regenerate icons with:

```powershell
python tools/generate_icon.py
```
