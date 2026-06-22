# DRS — Decision Review System

LBW assistance for club cricket. Play recorded video at full speed, pause on a delivery, and inspect ball/pitch/impact overlays.

**Full documentation: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)**

## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python main.py
python main.py --video mymatch.mp4
python main.py --live
```

## Controls (also shown on screen)

| Key | Action |
|-----|--------|
| Space | Play / Pause (analysis runs when paused) |
| A / D | Step 5 frames |
| J / K | Step 1 frame |
| Home / End | Start / end |
| R | Restart |
| B | Scrub buffer |
| S | Save clip |
| Q | Quit |

## License

MIT — see [LICENSE](LICENSE).
