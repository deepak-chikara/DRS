"""Multi-view ball position fusion on pitch plane."""

from __future__ import annotations

from dataclasses import dataclass

from drs.fusion.calibration import PitchCalibration, pixel_to_pitch


@dataclass
class FusedBall:
    pitch_x: float
    pitch_y: float
    confidence: float
    sources: list[str]
    uncertain: bool


class MultiViewFusion:
    """Fuse ball detections from leg-side and straight cameras."""

    def __init__(self, calibration: PitchCalibration | None = None):
        self.calibration = calibration
        self.leg_weight = 0.4
        self.straight_weight = 0.6

    def fuse(
        self,
        leg_ball: tuple[int, int, float, str] | None,
        straight_ball: tuple[int, int, float, str] | None,
        leg_camera: str = "leg_side",
        straight_camera: str = "straight_side",
    ) -> FusedBall | None:
        leg_pitch = None
        straight_pitch = None

        if self.calibration:
            if leg_ball and leg_ball[0] != 0:
                cam = self.calibration.cameras.get(leg_camera)
                if cam:
                    leg_pitch = pixel_to_pitch(cam.homography, leg_ball[0], leg_ball[1])
            if straight_ball and straight_ball[0] != 0:
                cam = self.calibration.cameras.get(straight_camera)
                if cam:
                    straight_pitch = pixel_to_pitch(cam.homography, straight_ball[0], straight_ball[1])

        if leg_pitch and straight_pitch:
            lx, ly = leg_pitch
            sx, sy = straight_pitch
            w_leg = self.leg_weight * (leg_ball[2] if leg_ball else 0.5)
            w_str = self.straight_weight * (straight_ball[2] if straight_ball else 0.5)
            total = w_leg + w_str
            if total > 0:
                px = (lx * w_leg + sx * w_str) / total
                py = (ly * w_leg + sy * w_str) / total
                return FusedBall(px, py, total / 2, ["leg_side", "straight_side"], False)

        if straight_pitch:
            return FusedBall(
                straight_pitch[0], straight_pitch[1],
                straight_ball[2] if straight_ball else 0.5, ["straight_side"], True,
            )
        if leg_pitch:
            return FusedBall(
                leg_pitch[0], leg_pitch[1],
                leg_ball[2] if leg_ball else 0.5, ["leg_side"], True,
            )

        if leg_ball and leg_ball[0] != 0:
            return FusedBall(float(leg_ball[0]), float(leg_ball[1]), leg_ball[2], [leg_ball[3]], True)
        if straight_ball and straight_ball[0] != 0:
            return FusedBall(
                float(straight_ball[0]), float(straight_ball[1]),
                straight_ball[2], [straight_ball[3]], True,
            )

        return None
