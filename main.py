"""
DRS Pro — Decision Review System for club cricket.

Run:  python main.py              (Qt desktop app)
      python main.py --legacy-opencv
      python main.py --video mymatch.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="DRS Pro - LBW video review")
    parser.add_argument("--video", "-v", default=None, help="Video file path")
    parser.add_argument("--config", "-c", default=None, help="Config file (default: user profile)")
    parser.add_argument("--live", action="store_true", help="Use USB camera (see config/match.yaml)")
    parser.add_argument("--legacy-opencv", action="store_true", help="Use legacy OpenCV window UI")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate stump lines (CLI)")
    parser.add_argument("--ground-id", "-g", default=None, help="Ground id for calibration")
    parser.add_argument("--version", action="store_true", help="Show version")
    args = parser.parse_args()

    if args.version:
        from drs import __version__
        print(__version__)
        return

    root = Path(__file__).resolve().parent

    if args.legacy_opencv or args.live or args.calibrate:
        _run_legacy_cli(args, root)
        return

    from drs.app.main import run
    from drs.config import CameraConfig, load_user_config, save_config
    from drs.paths import user_config_path

    config = load_user_config()
    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = root / video_path
        config.video_path = str(video_path)
        config.cameras["primary"] = CameraConfig("primary", "file", str(video_path), True)
        config.mode = "file"
        save_config(config, user_config_path())

    raise SystemExit(run(config))


def _run_legacy_cli(args, root: Path) -> None:
    from drs.config import CameraConfig, load_config, load_user_config
    from drs.pipeline import DRSPipeline
    from drs.ui.calibrate_stumps import run_calibration

    if args.config:
        cfg_path = root / args.config if not Path(args.config).is_absolute() else Path(args.config)
        config = load_config(cfg_path)
    elif args.live:
        config = load_config(root / "config" / "match.yaml")
    else:
        config = load_user_config()

    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = root / video_path
        config.video_path = str(video_path)
        config.cameras["primary"] = CameraConfig("primary", "file", str(video_path), True)
        config.mode = "file"

    if args.live:
        config.mode = "live"

    if args.calibrate:
        run_calibration(config, ground_id=args.ground_id)
        return

    DRSPipeline(config).run()


if __name__ == "__main__":
    main()
