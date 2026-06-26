# Changelog

## 1.0.0 — 2025-06-22

### Added

- **DRS Pro** desktop application (PySide6/Qt)
- Timeline, toolbar, menus, keyboard shortcuts
- First-run setup wizard and stump calibration tools
- Per-user settings in `%LOCALAPPDATA%\DRS`
- OUT / NOT OUT / REVIEW verdict assist
- Calibrated stump corridor lines on every frame
- Structured logging and session delivery logs
- Windows installer build scripts (PyInstaller + Inno Setup)
- Automated test suite and CI workflow
- EULA, privacy policy, and third-party notices

### Changed

- Detection runs on every frame (ball, batsman, pitch overlays)
- Version bumped from 0.2.0 prototype to 1.0.0 product

### Developer

- `python main.py` launches Qt app
- `python main.py --legacy-opencv` for OpenCV UI
- Headless `DRSEngine` for programmatic use
