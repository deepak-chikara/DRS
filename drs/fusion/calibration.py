"""Camera calibration, stump corridor geometry, and pitch-plane homography."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Three stumps span 9 inches (22.86 cm) across a 3.05 m pitch — matches pitch diagram.
STUMP_SET_WIDTH_FRAC = 22.86 / 305.0


@dataclass
class StumpPoints:
    """Four pixel points defining the LBW stump corridor (off and leg at each end)."""

    striker_off: tuple[int, int]
    striker_leg: tuple[int, int]
    bowler_off: tuple[int, int]
    bowler_leg: tuple[int, int]

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "striker_off": list(self.striker_off),
            "striker_leg": list(self.striker_leg),
            "bowler_off": list(self.bowler_off),
            "bowler_leg": list(self.bowler_leg),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> StumpPoints | None:
        if not data:
            return None
        required = ("striker_off", "striker_leg", "bowler_off", "bowler_leg")
        if not all(k in data for k in required):
            return None
        try:
            return cls(
                striker_off=(int(data["striker_off"][0]), int(data["striker_off"][1])),
                striker_leg=(int(data["striker_leg"][0]), int(data["striker_leg"][1])),
                bowler_off=(int(data["bowler_off"][0]), int(data["bowler_off"][1])),
                bowler_leg=(int(data["bowler_leg"][0]), int(data["bowler_leg"][1])),
            )
        except (IndexError, TypeError, ValueError):
            return None

    def is_valid(self) -> bool:
        return all(p != (0, 0) for p in (
            self.striker_off, self.striker_leg, self.bowler_off, self.bowler_leg
        ))


@dataclass
class CameraCalibration:
    name: str
    homography: np.ndarray  # 3x3 pixel -> pitch plane
    image_points: list[list[float]]
    pitch_points: list[list[float]]
    stump_points: StumpPoints | None = None


@dataclass
class PitchCalibration:
    ground_id: str
    pitch_length_m: float
    pitch_width_m: float
    cameras: dict[str, CameraCalibration]

    def save(self, path: str | Path) -> None:
        data = {
            "ground_id": self.ground_id,
            "pitch_length_m": self.pitch_length_m,
            "pitch_width_m": self.pitch_width_m,
            "cameras": {},
        }
        for name, cam in self.cameras.items():
            entry: dict = {
                "homography": cam.homography.tolist(),
                "image_points": cam.image_points,
                "pitch_points": cam.pitch_points,
            }
            if cam.stump_points is not None:
                entry["stump_points"] = cam.stump_points.to_dict()
            data["cameras"][name] = entry
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PitchCalibration | None:
        p = Path(path)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        cameras = {}
        for name, cam in data.get("cameras", {}).items():
            cameras[name] = CameraCalibration(
                name=name,
                homography=np.array(cam["homography"], dtype=np.float64),
                image_points=cam.get("image_points", []),
                pitch_points=cam.get("pitch_points", []),
                stump_points=StumpPoints.from_dict(cam.get("stump_points")),
            )
        return cls(
            ground_id=data.get("ground_id", "unknown"),
            pitch_length_m=data.get("pitch_length_m", 20.12),
            pitch_width_m=data.get("pitch_width_m", 3.05),
            cameras=cameras,
        )

    def get_stump_points(self, camera_name: str = "primary") -> StumpPoints | None:
        cam = self.cameras.get(camera_name)
        if cam and cam.stump_points and cam.stump_points.is_valid():
            return cam.stump_points
        for c in self.cameras.values():
            if c.stump_points and c.stump_points.is_valid():
                return c.stump_points
        return None


def compute_homography(image_points: list, pitch_points: list) -> np.ndarray:
    """Compute homography from pixel coords to normalized pitch plane (0-1)."""
    src = np.array(image_points, dtype=np.float32)
    dst = np.array(pitch_points, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def pixel_to_pitch(H: np.ndarray, x: int, y: int) -> tuple[float, float] | None:
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    return float(transformed[0][0][0]), float(transformed[0][0][1])


def pitch_to_pixel(H: np.ndarray, pitch_x: float, pitch_y: float) -> tuple[int, int] | None:
    H_inv = np.linalg.inv(H)
    pt = np.array([[[float(pitch_x), float(pitch_y)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H_inv)
    return int(transformed[0][0][0]), int(transformed[0][0][1])


def stump_x_at_y(p1: tuple[int, int], p2: tuple[int, int], y: float) -> float:
    """Linear interpolation of x along a stump line at image row y."""
    x1, y1 = p1
    x2, y2 = p2
    if y2 == y1:
        return float(x1)
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def corridor_bounds_at_y(stump_points: StumpPoints, y: float) -> tuple[float, float]:
    """Return (x_min, x_max) of the stump corridor at row y."""
    x_off = stump_x_at_y(stump_points.striker_off, stump_points.bowler_off, y)
    x_leg = stump_x_at_y(stump_points.striker_leg, stump_points.bowler_leg, y)
    return min(x_off, x_leg), max(x_off, x_leg)


def is_inside_corridor(x: int, y: int, stump_points: StumpPoints, margin_px: float = 0) -> bool:
    """True if (x, y) lies between the off and leg stump lines at that y."""
    x_min, x_max = corridor_bounds_at_y(stump_points, y)
    return (x_min - margin_px) <= x <= (x_max + margin_px)


def is_usable_homography(homography: np.ndarray | None) -> bool:
    """True when homography is a real pitch-plane calibration, not identity."""
    if homography is None or homography.shape != (3, 3):
        return False
    return not np.allclose(homography, np.eye(3), atol=1e-3)


def ground_y_at_pitch_length(ny: float, stump_points: StumpPoints) -> float:
    """Image row of the pitch surface at normalized length ny (0=striker, 1=bowler)."""
    striker_y = float(stump_points.striker_off[1])
    bowler_y = float(stump_points.bowler_off[1])
    return striker_y + ny * (bowler_y - striker_y)


def pixel_to_pitch_from_stumps(x: int, y: float, stump_points: StumpPoints) -> tuple[float, float]:
    """Map pixel coords to normalized pitch plane using locked stump corridor."""
    striker_y = float(stump_points.striker_off[1])
    bowler_y = float(stump_points.bowler_off[1])
    if abs(bowler_y - striker_y) < 1.0:
        ny = 0.5
    else:
        ny = (float(y) - striker_y) / (bowler_y - striker_y)
        ny = max(0.0, min(1.0, ny))

    x_min, x_max = corridor_bounds_at_y(stump_points, float(y))
    width = max(x_max - x_min, 1.0)
    rel = max(0.0, min(1.0, (float(x) - x_min) / width))
    nx = 0.5 + (rel - 0.5) * STUMP_SET_WIDTH_FRAC
    return max(0.0, min(1.0, nx)), ny


def pixel_to_pitch_normalized(
    x: int,
    y: int,
    *,
    frame_w: int,
    frame_h: int,
    homography: np.ndarray | None = None,
    stump_points: StumpPoints | None = None,
) -> tuple[float, float]:
    """Map pixel to normalized pitch coords: homography, then stumps, then frame fallback."""
    if is_usable_homography(homography):
        pt = pixel_to_pitch(homography, x, y)
        if pt is not None and -0.05 <= pt[0] <= 1.05 and -0.05 <= pt[1] <= 1.05:
            return max(0.0, min(1.0, pt[0])), max(0.0, min(1.0, pt[1]))

    if stump_points is not None and stump_points.is_valid():
        return pixel_to_pitch_from_stumps(x, y, stump_points)

    nx = max(0.0, min(1.0, x / max(frame_w, 1)))
    ny = max(0.0, min(1.0, 1.0 - y / max(frame_h, 1)))
    return nx, ny
