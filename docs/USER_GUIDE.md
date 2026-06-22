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

## 2. Quick start — recorded video

```powershell
python main.py
```

Or specify a video:

```powershell
python main.py --video lbw-2.mp4
```

### How it works
1. Video **plays at normal speed** (no slow processing while playing).
2. Press **Space** to **pause** on a delivery you want to check.
3. While paused, DRS runs analysis and shows overlays on the single result view:
   - Green = pitch outline
   - Red = batsman outline
   - Yellow lines = stump lines
   - Cyan dot = ball
   - Blue dot = pitch point (where ball bounced)
   - Red dot = impact point (possible LBW)
4. Use **J/K** or **A/D** to step frame-by-frame through the delivery.
5. Press **Space** again to resume playing.

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

```yaml
video: "lbw.mp4"        # default video file

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
| Green contour | Detected pitch area |
| Red contour | Detected batsman |
| Yellow vertical lines | Stump line estimate |
| Cyan dot | Current ball position |
| Blue "Pitch" marker | Ball pitch point (bounce) |
| Red "Impact" marker | Ball near batsman (possible contact) |
| "LBW?" text | Impact detected — review manually |

DRS **assists** review; it does not make automatic umpire decisions.

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
| Playback still slow | Ensure you are not paused; playing mode skips analysis |
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

## 10. Advanced

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
