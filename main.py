"""
DRS — Decision Review System for club cricket.

Run:  python main.py
      python main.py --video mymatch.mp4
      python main.py --live

Full guide: docs/USER_GUIDE.md
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="DRS - LBW video review")
    parser.add_argument("--video", "-v", default=None, help="Video file path")
    parser.add_argument("--config", "-c", default="config/default.yaml", help="Config file")
    parser.add_argument("--live", action="store_true", help="Use USB camera (see config/match.yaml)")
    args = parser.parse_args()

    from drs.config import CameraConfig, load_config
    from drs.pipeline import DRSPipeline

    root = Path(__file__).resolve().parent
    cfg_path = root / args.config if not Path(args.config).is_absolute() else Path(args.config)

    if args.live:
        cfg_path = root / "config" / "match.yaml"

    config = load_config(cfg_path)

    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = root / video_path
        config.video_path = str(video_path)
        config.cameras["primary"] = CameraConfig("primary", "file", str(video_path), True)
        config.mode = "file"

    if args.live:
        config.mode = "live"

    DRSPipeline(config).run()


if __name__ == "__main__":
    main()
