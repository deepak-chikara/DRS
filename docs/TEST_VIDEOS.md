# LBW test videos for DRS Pro

DRS Pro is tuned for **single side-on camera** footage (like a club ground camera behind the bowler). Broadcast DRS clips often use **multiple angles, ball-tracking graphics, and cuts** — they are still useful for manual testing, but expect lower CV accuracy than a fixed club angle.

---

## What makes a good test clip

| Prefer | Avoid |
|--------|--------|
| Side-on, whole pitch visible | Front-on TV replay only |
| Fixed camera (tripod) | Heavy zoom / slow-mo replays |
| Ball visible before release | Clips under 3 seconds |
| One delivery per clip | Montages with text overlays |
| 720p–1080p MP4 | Vertical phone video (works but harder) |

**Calibrate stumps once per camera angle** (Tools → Calibrate Stumps) before comparing OUT/NOT OUT across videos.

---

## Online sources (search & bookmark)

### 1. YouTube — best for bulk testing

Search these (pick **side-on** or **highlights with full pitch**):

| Search query | Typical content |
|--------------|-----------------|
| `club cricket lbw appeal side view` | Closest to your product |
| `village cricket lbw` | Amateur angle, fixed cam |
| `cricket lbw appeal slow motion side` | Clear ball + pad moment |
| `cricket coaching lbw front foot` | Often side-on drills |
| `local cricket lbw wicket` | Short club clips |

**Useful channels (browse their LBW / wicket playlists):**

- [Cricket Australia](https://www.youtube.com/@CricketAustralia) — match highlights (mixed angles)
- [England & Wales Cricket Board](https://www.youtube.com/@ECB) — DRS segments in highlights
- [ICC](https://www.youtube.com/@ICC) — international LBW reviews (often front-on + graphics)
- Club / league uploads — search your county + `"lbw"` or `"wicket"`

### 2. [LBW Test](https://lbwtest.com/) — decision practice

Random **LBW appeals** from real matches (YouTube embeds). Great for **umpire training**, not direct MP4 download. Use it to find interesting moments, then locate the same clip on YouTube for local download.

### 3. Broadcast / news (geo-restricted)

Often UK-only or subscription:

- [BBC Sport cricket videos](https://www.bbc.com/sport/cricket/videos)
- [Sky Sports cricket clips](https://www.skysports.com/cricket/video)
- [Wisden](https://www.wisden.com/) — embedded YouTube in articles

### 4. Your own footage (best validation)

- Phone on tripod behind bowler
- Club live stream recordings
- `%LOCALAPPDATA%\DRS\clips\` after **F9 DRS Call** in live mode

---

## Suggested test set (build locally)

Create a folder and aim for **variety**:

```
tests/fixtures/videos/
  out_clear_01.mp4          # obvious pad on middle stump line
  not_out_outside_01.mp4    # pad outside off/leg corridor
  review_marginal_01.mp4    # 50/50 line
  red_ball_club_01.mp4      # red ball, overcast
  white_ball_club_01.mp4    # white ball, bright sun
  short_clip_under_5s.mp4   # single delivery only
```

For each file, note in a spreadsheet:

- **Expected** (OUT / NOT OUT / REVIEW) for corridor line only
- **Camera** (side-on / other)
- **Ground preset** used after calibration

DRS Pro does **not** judge height, bat-pad, or snick — label expectations accordingly.

---

## Download clips for local testing

**Copyright:** Only download footage you have rights to use (your own, club with permission, or rights-free). Broadcast highlights are usually **personal test only**, not for redistribution in your installer/repo.

### Option A — Helper script (YouTube URL list)

1. Install [yt-dlp](https://github.com/yt-dlp/yt-dlp): `pip install yt-dlp`
2. Edit `tools/test_video_urls.txt` — one URL per line
3. Run:

```powershell
python tools/fetch_test_videos.py
```

Files land in `tests/fixtures/videos/`.

### Option B — Manual

1. Find a side-on LBW clip on YouTube  
2. Use yt-dlp or a browser extension you trust  
3. Save as `.mp4` in `tests/fixtures/videos/`  
4. **File → Open Video** in DRS Pro  

### Option C — Trim long matches

Use ffmpeg to cut a single delivery (~5–15 s):

```powershell
ffmpeg -ss 00:12:34 -i full_match.mp4 -t 12 -c copy tests/fixtures/videos/delivery_01.mp4
```

---

## Running DRS Pro on a test file

```powershell
python main.py --video tests/fixtures/videos/out_clear_01.mp4
```

Or **File → Open Video** in the app. Then:

1. **Tools → Calibrate Stumps** (once per camera angle)  
2. Pause at pad contact → check verdict banner + pitch diagram  
3. Optional: **Tools → Analyze Delivery with AI** (Ollama)

---

## Known limitations with online TV clips

| TV / DRS clip issue | Effect on DRS Pro |
|---------------------|-------------------|
| Front-on camera | Stump corridor mapping weak — calibrate or use side-on |
| Replays with graphics | Ball detection noise |
| Ultra slow-motion | Motion blur, odd HSV |
| Multiple camera cuts | Tracker resets each cut |

For regression testing without any online video, use synthetic frames: `python -m pytest tests/test_golden_frame.py`.

---

## Starter URL list

Add working side-on URLs you find to `tools/test_video_urls.txt`. Examples to **search** (do not commit copyrighted files to git):

- YouTube: `club cricket lbw appeal`
- YouTube: `cricket lbw side view full pitch`
- YouTube: `village cricket wicket lbw`

When you find a good clip, paste the URL into the txt file and run the fetch script.
