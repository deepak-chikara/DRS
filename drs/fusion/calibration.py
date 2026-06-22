"""Camera calibration and pitch-plane homography."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    name: str
    homography: np.ndarray  # 3x3 pixel -> pitch plane
    image_points: list[list[float]]
    pitch_points: list[list[float]]


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
            data["cameras"][name] = {
                "homography": cam.homography.tolist(),
                "image_points": cam.image_points,
                "pitch_points": cam.pitch_points,
            }
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
            )
        return cls(
            ground_id=data.get("ground_id", "unknown"),
            pitch_length_m=data.get("pitch_length_m", 20.12),
            pitch_width_m=data.get("pitch_width_m", 3.05),
            cameras=cameras,
        )


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
