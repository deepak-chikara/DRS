# DRS Pro — Deployment Guide

Checklist for club match-day deployment on Windows.

## Hardware

| Item | Recommendation |
|------|----------------|
| Camera | USB webcam or phone RTSP, **1080p** preferred |
| Mount | Behind bowler, elevated 2–3 m, centered on wicket |
| PC | Windows 10/11, 8 GB RAM minimum; GPU optional for hybrid mode |
| Lighting | Avoid strong backlight; overcast or floodlit evenings work best |

## One-time setup per ground

1. Install DRS Pro (installer or `pip install -r requirements.txt`).
2. **File → Start Live Match** (or open a sample video).
3. **Tools → Calibrate Stumps** — click four stump bases (striker off/leg, bowler off/leg).
4. **Tools → Settings** — tune ball HSV if color detection misses the ball; save as ground preset.
5. Optional: install [Ollama](https://ollama.com) and enable AI Review in Settings.

## Live match day

1. **File → Start Live Match** — pick ground ID, camera, hybrid detection, scale **0.35–0.5** on laptops.
2. Confirm stump corridor lines on video.
3. Press **F9** (DRS Call) at LBW moments — clip saves to `%LOCALAPPDATA%\DRS\clips\`.
4. **File → Browse Clips** to review saved deliveries.
5. Use **B** to scrub the ring buffer during live play; **Home/End** to jump in file review.

## Detection modes

| Mode | Use when |
|------|----------|
| `color` | Recorded video, good ball visibility, fastest |
| `hybrid` | Live camera; YOLO + color + Kalman (downloads YOLO once) |
| `yolo` | GPU available; generic sports-ball model |

## Product scope (club messaging)

DRS Pro provides **line assist** — pad on stump corridor from your camera. It does **not** adjudicate height, bat-pad, snick, or pitching outside off. See Help → About.

## Building the installer

```powershell
pip install pyinstaller
pyinstaller build/drs.spec
# Inno Setup 6:
iscc build/installer.iss
```

Output: `dist/installer/DRS-Pro-Setup-1.0.0.exe`

## AGPL (Ultralytics YOLO)

Hybrid/YOLO modes use Ultralytics YOLO (AGPL-3.0). Before commercial distribution, review `docs/THIRD_PARTY_NOTICES.md` and comply with source-offer requirements if you modify or bundle AGPL components.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Low FPS live | Lower detection scale in Settings (0.35) |
| Ball not tracked | Tune ball HSV in Settings; try hybrid mode |
| Wrong diagram line | Re-calibrate stumps for this ground |
| Ollama offline | `ollama serve` + `ollama pull llama3.2` |
| SmartScreen warning | Optional code signing on the installer EXE |
