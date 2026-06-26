"""Interactive stump corridor calibration UI."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drs.config import DRSConfig
from drs.fusion.calibration import (
    CameraCalibration,
    PitchCalibration,
    StumpPoints,
    compute_homography,
)
from drs.ui.stumps import draw_stump_corridor

STUMP_LABELS = [
    "striker_off",
    "striker_leg",
    "bowler_off",
    "bowler_leg",
]

PITCH_CORNERS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
]

CORNER_LABELS = [
    "bowler_end_left",
    "bowler_end_right",
    "striker_end_left",
    "striker_end_right",
]


class StumpCalibrationUI:
    def __init__(self, image: np.ndarray, title: str = "Calibrate stumps"):
        self.image = image.copy()
        self.display = image.copy()
        self.points: list[tuple[int, int]] = []
        self.window = title
        self._labels = STUMP_LABELS

    def _mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            cv2.circle(self.display, (x, y), 6, (0, 255, 0), -1)
            label = self._labels[len(self.points) - 1]
            cv2.putText(self.display, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            if len(self.points) == 4:
                sp = StumpPoints(*self.points)
                self.display = draw_stump_corridor(self.image.copy(), sp, fill=True)
                for i, pt in enumerate(self.points):
                    cv2.circle(self.display, pt, 6, (0, 255, 0), -1)
                    cv2.putText(self.display, self._labels[i], (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imshow(self.window, self.display)

    def run(self) -> list[tuple[int, int]]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._mouse_callback)
        cv2.imshow(self.window, self.display)
        print(f"Click 4 stump bases in order: {STUMP_LABELS}")
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


class CornerCalibrationUI:
    def __init__(self, image: np.ndarray, camera_name: str):
        self.image = image.copy()
        self.display = image.copy()
        self.points: list[tuple[int, int]] = []
        self.window = f"Calibrate corners - {camera_name}"
        self._labels = CORNER_LABELS

    def _mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            cv2.circle(self.display, (x, y), 5, (0, 255, 0), -1)
            label = self._labels[len(self.points) - 1]
            cv2.putText(self.display, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow(self.window, self.display)

    def run(self) -> list[tuple[int, int]]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._mouse_callback)
        cv2.imshow(self.window, self.display)
        print(f"Click 4 pitch corners: {CORNER_LABELS}")
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


def read_calibration_frame(video_path: str, frame_index: int = 0) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Cannot read calibration frame.")
        return None
    return frame


def calibrate_stumps_on_frame(
    frame: np.ndarray,
    *,
    camera_name: str = "primary",
    include_corners: bool = False,
    existing: CameraCalibration | None = None,
) -> CameraCalibration | None:
    """Run stump click UI; optionally collect pitch corners first."""
    homography = existing.homography if existing is not None else np.eye(3)
    image_points = existing.image_points if existing else []
    pitch_points = existing.pitch_points if existing else PITCH_CORNERS

    if include_corners:
        corner_ui = CornerCalibrationUI(frame, camera_name)
        corners = corner_ui.run()
        if len(corners) != 4:
            print("Need exactly 4 corner points.")
            return None
        homography = compute_homography(corners, PITCH_CORNERS)
        image_points = [[p[0], p[1]] for p in corners]
        pitch_points = PITCH_CORNERS

    stump_ui = StumpCalibrationUI(frame, f"Calibrate stumps - {camera_name}")
    stump_pts = stump_ui.run()
    if len(stump_pts) != 4:
        print("Need exactly 4 stump points.")
        return None

    return CameraCalibration(
        name=camera_name,
        homography=homography,
        image_points=image_points,
        pitch_points=pitch_points,
        stump_points=StumpPoints(*stump_pts),
    )


def save_pitch_calibration(
    cam: CameraCalibration,
    *,
    ground_id: str,
    output_path: Path,
) -> Path:
    cal = PitchCalibration(
        ground_id=ground_id,
        pitch_length_m=20.12,
        pitch_width_m=3.05,
        cameras={cam.name: cam},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cal.save(output_path)
    return output_path


def run_calibration(
    config: DRSConfig,
    *,
    ground_id: str | None = None,
    output_path: Path | None = None,
    video_path: str | None = None,
    include_corners: bool = False,
    continue_into_drs: bool = False,
) -> Path | None:
    """Inline calibration from config video path."""
    root = _ROOT
    gid = ground_id or config.ground_id or "default"
    out = output_path or (root / "config" / "calibration" / f"{gid}.json")
    if not out.is_absolute():
        out = root / out

    vpath = video_path or config.video_path
    frame = read_calibration_frame(vpath)
    if frame is None:
        return None

    existing_cal = None
    if config.calibration_file:
        existing_path = Path(config.calibration_file)
        if not existing_path.is_absolute():
            existing_path = root / existing_path
        existing = PitchCalibration.load(existing_path)
        if existing:
            existing_cal = existing.cameras.get("primary") or next(iter(existing.cameras.values()), None)

    cam = calibrate_stumps_on_frame(
        frame,
        camera_name="primary",
        include_corners=include_corners,
        existing=existing_cal,
    )
    if cam is None:
        return None

    saved = save_pitch_calibration(cam, ground_id=gid, output_path=out)
    print(f"Calibration saved to {saved}")
    config.calibration_file = str(saved)
    config.ground_id = gid

    if continue_into_drs:
        answer = input("Continue into DRS? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            return saved
    return saved
