# DRS — User Guide

Decision Review System for club cricket. Review recorded match footage or live camera feeds, pause on a delivery, and inspect ball/pitch/impact overlays.

---

## 1. Setup (one time)

### Requirements
- Windows 10/11
- Python 3.10 or newer
- A cricket video file (`.mp4`) or USB webcam

### Install

```powershell
cd DRS
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place your video in the `DRS` folder (e.g. `lbw.mp4`) or pass the path with `--video`.

---

## 2. Quick start — DRS Pro desktop app

```powershell
python main.py
```

This launches the **DRS Pro** Qt desktop application with:

- Menu bar (File, Tools, Help)
- Video panel with stump corridor overlays
- Timeline slider and transport controls
- Setup wizard on first run (calibrate stumps per ground)

### Open a video

**File → Open Video** or use the toolbar. Supported: `.mp4`, `.avi`, `.mov`.

### Legacy OpenCV UI (developers)

```powershell
python main.py --legacy-opencv
```

### Command-line video path

```powershell
python main.py --video lbw-2.mp4
```

(Opens Qt app; set video in File → Open if not auto-loaded.)

## 2b. Quick start — recorded video (legacy)

```powershell
python main.py
```

Or specify a video:

```powershell
python main.py --video lbw-2.mp4
```

### How it works
1. Video plays with **stump lines**, **ball tracking**, and **batsman detection** on every frame.
2. Yellow perspective lines = stump corridor (off-stump to off-stump, leg-stump to leg-stump). Lines stay **fixed** once set — they do not move when the batsman moves.
3. When the ball hits the pad inside the corridor, DRS shows **OUT**; outside = **NOT OUT**; uncertain = **REVIEW**.
4. Press **Space** to pause and step frame-by-frame with **J/K** or **A/D**.
5. Press **R** to restart the video and reset the delivery state.

**Important:** OUT/NOT OUT is an **assist** for club review — not an official umpire decision.

### Calibrate stump lines (do once per ground)

When you move the camera to a new ground, calibrate the stump corridor:

**Option A — inline at app start:**
```powershell
python main.py --calibrate --ground-id riverside
python main.py --calibrate --ground-id riverside --continue-after-calibrate
```

Click in order: **striker off**, **striker leg**, **bowler off**, **bowler leg** stump bases.  
Saved to `config/calibration/riverside.json`.

**Option B — standalone tool:**
```powershell
python tools/calibrate_pitch.py --video lbw.mp4 --ground-id riverside --stumps-only
```

Then add to `config/default.yaml`:
```yaml
ground_id: "riverside"
calibration_file: "config/calibration/riverside.json"
```

Without calibration, DRS locks stump lines from the **first frame** of the video (before batsman movement). Press **R** to restart and re-lock. For best accuracy, calibrate once per ground (see above).

---

## 3. Keyboard controls

All controls are listed on the right side of the screen.

| Key | Action |
|-----|--------|
| **Space** | Play / Pause |
| **A** or **D** | Step backward / forward 5 frames (when paused) |
| **J** or **K** | Step backward / forward 1 frame (when paused) |
| **Home** | Jump to start of video |
| **End** | Jump to end of video |
| **R** | Restart from beginning |
| **B** | Scrub the last ~30 seconds from memory buffer |
| **Esc** | Exit buffer scrub mode |
| **S** | Save a clip to `sessions/clips/` |
| **Q** | Quit |

**Typical DRS review workflow**
1. Play the video.
2. Pause (**Space**) when you see a possible LBW.
3. Step with **J/K** to find the exact impact frame.
4. Read the overlays (impact point, pitch point, "LBW?" label).
5. Press **Space** to continue or **S** to save the clip.

---

## 4. Configuration

Edit `config/default.yaml`:

**File mode** uses the `video:` key only. Camera settings (`cameras:`) are for **live mode** — see `config/match.yaml`.

```yaml
video: "lbw.mp4"        # default video file
ground_id: "default"
calibration_file: ""    # e.g. config/calibration/riverside.json

detection:
  mode: color             # keep as color for recorded video (fast)

ball:
  hsv: ...                # tune if ball is not detected

batsman:
  rgb_lower: ...          # tune if batsman outline is wrong
  rgb_upper: ...

buffer_seconds: 30        # how much footage B (buffer) can scrub
```

### Tuning detection

If ball or batsman are not detected correctly on your footage:

**Batsman colours** — run the tuning tool:
```powershell
python batsman.py
```
Adjust trackbars until the batsman is highlighted, press **S** to print values, then copy them into `config/default.yaml` under `batsman`.

**Ball colour** — edit `ball.hsv` in `config/default.yaml`, or test with:
```powershell
python ball_detect.py
```

---

## 5. Live camera (club ground)

1. Connect USB camera(s).
2. Edit `config/match.yaml` — set `source: 0` to your camera index.
3. Run:
```powershell
python main.py --live
```
Live mode runs detection continuously (slower than file playback). Use **Space** to pause and review.

For two cameras, set `secondary.enabled: true` and the second camera index.

---

## 6. What the overlays mean

| Overlay | Meaning |
|---------|---------|
| Yellow perspective lines | Stump corridor (calibrated or locked from first frame) |
| Green contour | Detected pitch area |
| Red contour | Detected batsman |
| Cyan dot | Current ball position |
| Blue "Pitch" marker | Ball pitch point (bounce) |
| Red "Impact" marker | Ball near batsman (possible contact) |
| **OUT** (red) | Pad contact inside stump corridor |
| **NOT OUT** (green) | Pad contact outside stump corridor |
| **REVIEW** (orange) | Uncertain — tune detection or calibrate stumps |
| "LBW?" text | Impact suspected before verdict computed |

DRS **assists** review; it does not make automatic ICC umpire decisions.

---

## 7. Output files

| Location | Contents |
|----------|----------|
| `sessions/clips/` | MP4 clips saved with **S** |
| `sessions/` | Session logs (if enabled) |

---

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| Video won't open | Check file path; use `--video path\to\file.mp4` |
| Playback slow | Lower `performance.detection_scale` in config (e.g. 0.35) |
| Stump lines wrong | Run `python main.py --calibrate --ground-id your_ground` |
| No OUT/NOT OUT shown | Pause on pad frame; calibrate stumps; tune ball/batsman |
| No ball detected | Tune `ball.hsv` in config |
| No batsman detected | Run `python batsman.py` and update config |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Camera not found | Try `source: 0`, `1`, or `2` in `config/match.yaml` |

---

## 9. Project layout

```
DRS/
  main.py              ← start here
  config/
    default.yaml       ← video + detection settings
    match.yaml         ← live camera settings
  docs/
    USER_GUIDE.md      ← this file
  ball_detect.py       ← ball detection module
  batsman.py           ← batsman detection + tuning
  pitch.py             ← pitch detection module
  drs/                 ← application code
  requirements.txt
```

---

## 11. Building the Windows installer

For vendors distributing DRS Pro:

```powershell
pip install pyinstaller
pyinstaller build/drs.spec
```

Compile `build/installer.iss` with [Inno Setup 6](https://jrsoftware.org/isinfo.php).

Output: `dist/installer/DRS-Pro-Setup-1.0.0.exe`

Customer data lives in `%LOCALAPPDATA%\DRS` — not the install folder.

## 12. Advanced

**Different config file:**
```powershell
python main.py --config config\mysettings.yaml
```

**Hybrid / YOLO detection (live only, slower, needs ultralytics):**
Set in `config/match.yaml`:
```yaml
detection:
  mode: hybrid
```

**Requirements:** `opencv-python`, `numpy`, `cvzone`, `pyyaml`. Optional: `ultralytics` for live hybrid mode.
