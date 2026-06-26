"""Pitch calibration tool — click pitch corners and stump bases on a still frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from drs.fusion.calibration import CameraCalibration, PitchCalibration, compute_homography
from drs.ui.calibrate_stumps import (
    CornerCalibrationUI,
    calibrate_stumps_on_frame,
    read_calibration_frame,
)


def calibrate_camera(
    video_path: str,
    camera_name: str,
    frame_index: int = 0,
    *,
    stumps_only: bool = False,
) -> CameraCalibration | None:
    frame = read_calibration_frame(video_path, frame_index)
    if frame is None:
        return None

    if stumps_only:
        return calibrate_stumps_on_frame(frame, camera_name=camera_name, include_corners=False)

    corner_ui = CornerCalibrationUI(frame, camera_name)
    corners = corner_ui.run()
    if len(corners) != 4:
        print("Need exactly 4 corner points.")
        return None

    H = compute_homography(corners, PITCH_CORNERS)
    existing = CameraCalibration(
        name=camera_name,
        homography=H,
        image_points=[[p[0], p[1]] for p in corners],
        pitch_points=PITCH_CORNERS,
    )
    return calibrate_stumps_on_frame(frame, camera_name=camera_name, include_corners=False, existing=existing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate pitch homography and stump lines for DRS")
    parser.add_argument("--video", help="Video file for single-camera calibration")
    parser.add_argument("--leg-video", help="Video file for leg-side camera (dual setup)")
    parser.add_argument("--straight-video", help="Video file for straight camera (optional)")
    parser.add_argument("--camera", default="primary", help="Camera name in calibration JSON")
    parser.add_argument("--stumps-only", action="store_true", help="Click stump points only (skip corners)")
    parser.add_argument("--ground-id", default="club_ground", help="Ground identifier")
    parser.add_argument("--output", default="config/calibration/club_ground.json", help="Output JSON path")
    args = parser.parse_args()

    video = args.video or args.leg_video
    if not video:
        parser.error("Provide --video or --leg-video")

    cameras: dict[str, CameraCalibration] = {}
    cam = calibrate_camera(video, args.camera, stumps_only=args.stumps_only)
    if cam:
        cameras[cam.name] = cam

    if args.straight_video:
        straight = calibrate_camera(args.straight_video, "straight_side", stumps_only=args.stumps_only)
        if straight:
            cameras["straight_side"] = straight

    if not cameras:
        print("No cameras calibrated.")
        return

    cal = PitchCalibration(
        ground_id=args.ground_id,
        pitch_length_m=20.12,
        pitch_width_m=3.05,
        cameras=cameras,
    )
    out = Path(args.output)
    if not out.is_absolute():
        out = _ROOT / out
    cal.save(out)
    print(f"Calibration saved to {out}")
    print(f"Add to config/default.yaml: calibration_file: \"{out.relative_to(_ROOT).as_posix()}\"")


if __name__ == "__main__":
    main()
