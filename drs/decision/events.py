"""Decision logic for pitch, impact, and pad detection."""

from __future__ import annotations

import cv2

from drs.config import DRSConfig
from drs.state import DRSState


def ball_pitch_pad(
    x: int,
    x_prev: int,
    prev_x_diff: int,
    y: int,
    y_prev: int,
    prev_y_diff: int,
    bat_leg: int,
) -> tuple[str, int, int]:
    """Classify ball motion as Pad, Pitch, or Motion (from original main.py)."""
    if x_prev == 0 and y_prev == 0:
        return "Motion", 0, 0

    if abs(x - x_prev) > 3 * abs(prev_x_diff) and abs(prev_x_diff) > 0:
        if y < bat_leg:
            return "Pad", x - x_prev, y - y_prev

    if y - y_prev < 0 and prev_y_diff > 0:
        if y < bat_leg:
            return "Pad", x - x_prev, y - y_prev
        return "Pitch", x - x_prev, y - y_prev

    return "Motion", x - x_prev, y - y_prev


class EventDetector:
    """Detect pitch point, impact, and pad events from per-frame state."""

    def __init__(self, config: DRSConfig):
        self.config = config
        self._delivery_motion_frames = 0
        self._delivery_latched = False

    def reset(self) -> None:
        self._delivery_motion_frames = 0
        self._delivery_latched = False

    def _ball_speed(self, state: DRSState) -> float:
        b = state.ball
        return float((b.prev_x_diff ** 2 + b.prev_y_diff ** 2) ** 0.5)

    def _delivery_in_progress(self, state: DRSState) -> bool:
        """True once the ball is moving toward the batsman (post-release)."""
        if self._delivery_latched:
            return True

        b = state.ball
        min_motion = self.config.delivery_motion_min_px
        if b.x == 0 and b.y == 0:
            return False

        toward_batsman = b.prev_y_diff >= min_motion
        if self._ball_speed(state) >= min_motion and toward_batsman:
            self._delivery_motion_frames += 1
        elif self._delivery_motion_frames > 0:
            self._delivery_motion_frames -= 1

        if self._delivery_motion_frames >= self.config.delivery_motion_frames:
            self._delivery_latched = True
        return self._delivery_latched

    def process_frame(
        self,
        state: DRSState,
        pitch_contours: list,
        batsman_contours: list,
    ) -> None:
        x = state.ball.x
        y = state.ball.y
        bat_leg = state.bat_leg
        min_motion = self.config.delivery_motion_min_px

        motion_class, _, _ = ball_pitch_pad(
            x,
            state.ball.x_prev,
            state.ball.prev_x_diff,
            y,
            state.ball.y_prev,
            state.ball.prev_y_diff,
            bat_leg,
        )
        state.last_motion_class = motion_class
        if motion_class == "Pad":
            state.pad_detected = True

        delivery_active = self._delivery_in_progress(state)

        if state.pitch_point is None and delivery_active and x != 0 and y != 0:
            if state.ball.prev_y_diff < min_motion:
                state.pitch_counter = 0
            else:
                for cnt in pitch_contours:
                    if cv2.contourArea(cnt) > self.config.pitch_area_min:
                        inside = cv2.pointPolygonTest(cnt, (x, y), False)
                        if inside >= 0:
                            state.pitch_counter += 1
                        else:
                            state.pitch_counter = 0
                        if state.pitch_counter > self.config.pitch_stable_frames:
                            state.pitch_point = (x, y)
                        break

        current_bat_leg = 10000
        for cnt in batsman_contours:
            if cv2.contourArea(cnt) > self.config.batsman_area_min:
                if min(cnt[:, :, 1]) < y and y != 0:
                    bat_leg_candidate = max(cnt[:, :, 1])
                    if bat_leg_candidate < current_bat_leg:
                        current_bat_leg = bat_leg_candidate
        state.bat_leg = current_bat_leg

        if (
            not state.impact_locked
            and delivery_active
            and state.pitch_point is not None
            and x != 0
            and y != 0
            and state.bat_leg != 10000
            and y >= state.pitch_point[1] - self.config.impact_batleg_px
        ):
            for cnt in batsman_contours:
                if cv2.contourArea(cnt) > self.config.batsman_area_min:
                    dist = cv2.pointPolygonTest(cnt, (x, y), True)
                    if dist >= -self.config.impact_distance_px and abs(y - state.bat_leg) < self.config.impact_batleg_px:
                        state.impact_point = (x, y)
                        state.impact_locked = True
                        state.lbw_detected = True
                        state.delivery_count += 1
                        break

        if state.lbw_detected or state.pad_detected:
            state.lbw_detected = state.lbw_detected or state.pad_detected
