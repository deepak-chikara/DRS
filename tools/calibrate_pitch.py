"""Pitch calibration tool — click pitch corners and stumps on a still frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from drs.fusion.calibration import CameraCalibration, PitchCalibration, compute_homography

POINT_LABELS = [
    "bowler_end_left",
    "bowler_end_right",
    "striker_end_left",
    "striker_end_right",
]

PITCH_CORNERS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
]


class CalibrationUI:
    def __init__(self, image, camera_name: str):
        self.image = image.copy()
        self.display = image.copy()
        self.camera_name = camera_name
        self.points: list[tuple[int, int]] = []
        self.window = f"Calibrate - {camera_name}"

    def _mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            cv2.circle(self.display, (x, y), 5, (0, 255, 0), -1)
            label = POINT_LABELS[len(self.points) - 1] if len(self.points) <= 4 else "?"
            cv2.putText(self.display, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow(self.window, self.display)

    def run(self) -> list[tuple[int, int]]:
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._mouse_callback)
        cv2.imshow(self.window, self.display)
        print(f"Click 4 pitch corners for {self.camera_name}: {POINT_LABELS}")
        print("Press Enter when done, r to reset, q to quit.")
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 13 and len(self.points) == 4:
                break
            if key == ord("r"):
                self.points.clear()
                self.display = self.image.copy()
                cv2.imshow(self.window, self.display)
            if key == ord("q"):
                break
        cv2.destroyWindow(self.window)
        return self.points


def calibrate_camera(video_path: str, camera_name: str, frame_index: int = 0) -> CameraCalibration | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Cannot read frame.")
        return None

    ui = CalibrationUI(frame, camera_name)
    points = ui.run()
    if len(points) != 4:
        print("Need exactly 4 points.")
        return None

    H = compute_homography(points, PITCH_CORNERS)
    return CameraCalibration(
        name=camera_name,
        homography=H,
        image_points=[[p[0], p[1]] for p in points],
        pitch_points=PITCH_CORNERS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate pitch homography for DRS cameras")
    parser.add_argument("--leg-video", required=True, help="Video file for leg-side camera")
    parser.add_argument("--straight-video", help="Video file for straight camera (optional)")
    parser.add_argument("--ground-id", default="club_ground", help="Ground identifier")
    parser.add_argument("--output", default="config/calibration/club_ground.json", help="Output JSON path")
    args = parser.parse_args()

    cameras = {}
    leg = calibrate_camera(args.leg_video, "leg_side")
    if leg:
        cameras["leg_side"] = leg

    if args.straight_video:
        straight = calibrate_camera(args.straight_video, "straight_side")
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


if __name__ == "__main__":
    main()
