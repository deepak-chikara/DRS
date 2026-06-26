"""Export animated pitch diagram MP4 from a recorded video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drs.analysis.video_trajectory import extract_trajectories_from_video
from drs.config import load_config
from drs.ui.pitch_diagram import export_combined_diagram_video, tracking_verdict_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Export animated pitch diagram from match video")
    parser.add_argument("--video", "-v", default="lbw.mp4", help="Input video path")
    parser.add_argument("--output", "-o", default=None, help="Output MP4 path")
    parser.add_argument("--delivery", "-d", type=int, default=0, help="Delivery index (0-based)")
    parser.add_argument("--config", "-c", default="config/default.yaml", help="DRS config file")
    parser.add_argument("--fps", type=float, default=24.0, help="Diagram animation FPS")
    args = parser.parse_args()

    root = _ROOT
    video = Path(args.video)
    if not video.is_absolute():
        video = root / video
    if not video.is_file():
        print(f"Video not found: {video}")
        sys.exit(1)

    cfg_path = root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(cfg_path)
    config.video_path = str(video)

    print(f"Extracting trajectory from {video.name}...")
    deliveries = extract_trajectories_from_video(video, config)
    if not deliveries:
        print("No ball trajectory detected. Tune ball.hsv in config or use hybrid detection.")
        sys.exit(1)

    idx = min(args.delivery, len(deliveries) - 1)
    delivery = deliveries[idx]
    print(f"Delivery {idx}: {len(delivery.pitch_points)} tracking points")
    label = tracking_verdict_label(len(delivery.pitch_points))
    if label:
        print(f"  {label}")
    if delivery.verdict:
        print(f"  Verdict: {delivery.verdict} — {delivery.verdict_reason}")

    out = Path(args.output) if args.output else root / "output" / f"{video.stem}_diagram_d{idx}.mp4"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    export_combined_diagram_video(
        delivery.pitch_points,
        delivery.pixel_points,
        str(out),
        fps=args.fps,
        frame_h=delivery.frame_h,
        pitch_bounce=delivery.pitch_bounce,
        impact=delivery.impact,
    )
    print(f"Saved animated diagram: {out}")


if __name__ == "__main__":
    main()
