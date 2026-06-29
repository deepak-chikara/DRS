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

### Start a live match (Qt app — recommended)

1. Connect USB camera(s).
2. Edit [`config/match.yaml`](config/match.yaml) — set `source: 0` to your camera index.
3. Run `python main.py` → **File → Start Live Match** (or `python main.py --live`).

**What happens automatically:**

- **Full match recording** to `%LOCALAPPDATA%\DRS\matches\{ground}_{timestamp}.mp4`
- **60-second ring buffer** for instant DRS clips
- **Pitch diagram** panel (top-down ball path animation)
- Status bar shows `REC mm:ss — filename`

### DRS call during live play

When an LBW review is needed:

1. Press **F9** or toolbar **DRS Call** immediately when the ball is bowled.
2. DRS saves a clip: **12s before** + **8s after** the call (configurable in `match.yaml`).
3. A dialog offers to **open the clip** for frame-by-frame review (same as recorded video).
4. Session JSON is saved under `%LOCALAPPDATA%\DRS\sessions\drs_call_*.json`.

### Legacy OpenCV live mode

```powershell
python main.py --live --legacy-opencv
```

### Live config (`config/match.yaml`)

```yaml
buffer_seconds: 60
recording:
  enabled: true
  segment_minutes: 45
  width: 1280
clip:
  pre_roll_seconds: 12
  post_roll_seconds: 8
diagram:
  enabled: true
```

For two cameras, set `secondary.enabled: true` and the second camera index.

### Pitch diagram (animated fallback)

When live ball tracking is sparse, the **Pitch diagram** dock shows a top-down Hawk-Eye-style path:

| Points tracked | Display |
|----------------|---------|
| 5+ | Solid path + verdict assist |
| 2–4 | Dashed path + **REVIEW** |
| 0–1 | Message: use recorded clip |

Calibrate stumps per ground for best diagram accuracy.

### Export diagram from sample / recorded video

Process any video (including `lbw.mp4`) into an animated top-down MP4:

```powershell
python tools/export_pitch_diagram.py --video lbw.mp4 --output output/lbw_diagram.mp4
```

In the Qt app: open the video → **Tools → Export Diagram Video...** — saves an MP4 and updates the dock panel.

While playing a recorded video, the **Pitch diagram** dock on the right animates in sync with ball tracking (same as live mode).

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
| `%LOCALAPPDATA%\DRS\matches\` | Full live match MP4 recordings |
| `%LOCALAPPDATA%\DRS\clips\` | DRS call clips (auto on F9) |
| `%LOCALAPPDATA%\DRS\sessions\` | DRS call session JSON |
| `sessions/clips/` | Manual clips saved with **S** |

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

**Requirements:** `opencv-python`, `numpy`, `cvzone`, `pyyaml`, `httpx`. Optional: `ultralytics` for live hybrid mode; Ollama for AI advisory.

---

## 13. AI Advisory (Ollama)

DRS Pro includes an optional **AI Review** panel that uses a local [Ollama](https://ollama.com) model to explain decisions, score evidence quality, and resolve ambiguous REVIEW cases.

### Setup Ollama

1. Install Ollama from https://ollama.com
2. Pull a model (recommended for speed and JSON output):

```powershell
ollama pull llama3.2
```

3. Ensure Ollama is running (`ollama serve` starts automatically on Windows).

### Enable in DRS

1. In the app: **Tools → Settings** (or **Tools → AI Settings**)
2. Check **Enable AI Review panel and Ollama advisory**
3. Set model to `llama3.2` (or your pulled model)
4. Click **Test Ollama** — should say Connected
5. Click **Save**

Alternatively edit `%LOCALAPPDATA%\DRS\settings.yaml`:

```yaml
ai:
  enabled: true
  provider: ollama
  min_confidence_auto: 0.72
  resolve_review: true
  ollama:
    base_url: "http://127.0.0.1:11434"
    model: "llama3.2"
    timeout_seconds: 25
    temperature: 0.1
```

Restart DRS Pro. The **AI Review** dock shows Ollama connection status.

### How it works

- **Computer vision** remains primary: ball detection and stump corridor geometry produce OUT / NOT OUT / REVIEW.
- **Confidence scoring** downgrades weak OUT/NOT OUT calls to REVIEW when tracking or calibration is poor.
- **AI advisory** runs asynchronously (never blocks video playback). It receives structured delivery evidence and returns a recommended verdict with reasoning.
- AI may resolve **REVIEW → OUT/NOT OUT** when confidence is high; it **cannot silently override** a confident CV OUT/NOT OUT that it disagrees with.

### File and live mode

- **Tools → Analyze Delivery with AI** runs Ollama on the **current frame/delivery** (does not enable AI by itself).
- **Automatic:** When AI is enabled, analysis also runs when OUT/NOT OUT/REVIEW is issued during playback.
- **Live mode:** Press **DRS Call** (F9); when the clip is saved, AI review runs on that delivery.

### Alternative free models

```powershell
ollama pull qwen2.5:3b
```

Set `ai.ollama.model: "qwen2.5:3b"` in settings.
