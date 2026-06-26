"""Display and review UI helpers."""

from __future__ import annotations

import cv2
import numpy as np

from drs.config import DRSConfig
from drs.detectors.hybrid import HybridResult
from drs.fusion.calibration import StumpPoints
from drs.state import DRSState
from drs.ui.playback import ON_SCREEN_CONTROLS
from drs.ui.stumps import draw_stump_corridor


def match_panel_height(img: np.ndarray, target_height: int) -> np.ndarray:
    """Resize image to target height, preserving aspect ratio."""
    h, w = img.shape[:2]
    if h == target_height:
        return img
    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)


def draw_overlays(
    img: np.ndarray,
    result: HybridResult,
    state: DRSState,
    config: DRSConfig,
    stump_points: StumpPoints | None = None,
) -> np.ndarray:
    """Draw stump corridor, detection overlays, and verdict on a single result view."""
    out = img.copy()
    if stump_points is not None and stump_points.is_valid():
        out = draw_stump_corridor(out, stump_points, fill=True)

    for cnt in result.pitch_contours:
        if cv2.contourArea(cnt) > config.pitch_area_min:
            cv2.drawContours(out, cnt, -1, (0, 255, 0), 2)

    for cnt in result.batsman_contours:
        if cv2.contourArea(cnt) > config.batsman_area_min:
            cv2.drawContours(out, cnt, -1, (0, 0, 255), 2)

    if state.pitch_point is not None:
        cv2.circle(out, state.pitch_point, 8, (255, 0, 0), -1)
        cv2.putText(out, "Pitch", state.pitch_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if state.impact_point is not None:
        cv2.circle(out, state.impact_point, 10, (0, 0, 255), -1)
        cv2.putText(out, "Impact", state.impact_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if result.ball_x != 0 or result.ball_y != 0:
        cv2.circle(out, (result.ball_x, result.ball_y), 6, (0, 255, 255), -1)

    if state.last_motion_class != "Motion":
        cv2.putText(out, state.last_motion_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    if state.lbw_detected and not state.verdict:
        cv2.putText(out, "LBW?", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if state.verdict:
        colors = {
            "OUT": (0, 0, 255),
            "NOT OUT": (0, 200, 0),
            "REVIEW": (0, 165, 255),
        }
        color = colors.get(state.verdict, (255, 255, 255))
        cv2.putText(out, state.verdict, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        if state.verdict_reason:
            cv2.putText(out, state.verdict_reason[:55], (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return out


def render_frame(
    img: np.ndarray,
    result: HybridResult,
    state: DRSState,
    config: DRSConfig,
    stump_points: StumpPoints | None = None,
) -> np.ndarray:
    """Headless frame render for Qt (no sidebar)."""
    return draw_overlays(img, result, state, config, stump_points)


def draw_controls_panel(height: int, width: int = 260) -> np.ndarray:
    """Draw a permanent on-screen panel with key badges and actions."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)

    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (0, 200, 200), 1)
    cv2.putText(panel, "CONTROLS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    row_h = 34
    y = 48
    font = cv2.FONT_HERSHEY_SIMPLEX

    for key_label, action in ON_SCREEN_CONTROLS:
        if y + row_h > height - 8:
            break

        badge_w = max(52, 8 + len(key_label) * 11)
        badge_x1, badge_y1 = 10, y
        badge_x2, badge_y2 = 10 + badge_w, y + 24

        cv2.rectangle(panel, (badge_x1, badge_y1), (badge_x2, badge_y2), (50, 50, 50), -1)
        cv2.rectangle(panel, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 220, 220), 1)

        text_size = cv2.getTextSize(key_label, font, 0.45, 1)[0]
        key_x = badge_x1 + max(4, (badge_w - text_size[0]) // 2)
        cv2.putText(panel, key_label, (key_x, badge_y2 - 7), font, 0.45, (255, 255, 255), 1)

        action_x = badge_x2 + 8
        if action_x + 120 < width:
            cv2.putText(panel, action, (action_x, badge_y2 - 7), font, 0.4, (190, 190, 190), 1)
        else:
            cv2.putText(panel, action[:18], (10, badge_y2 + 14), font, 0.35, (190, 190, 190), 1)

        y += row_h

    return panel


def draw_status_banner(
    img: np.ndarray,
    playback_mode: str,
    *,
    frame_info: str = "",
) -> np.ndarray:
    """Draw mode and frame info banner at the bottom of the video area."""
    out = img.copy()
    banner_h = 32
    banner_y = out.shape[0] - banner_h
    cv2.rectangle(out, (0, banner_y), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    label = f"MODE: {playback_mode.upper()}"
    if frame_info:
        label += f"  |  {frame_info}"
    if playback_mode == "playing":
        label += "  |  Space = pause"
    cv2.putText(out, label, (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


def attach_controls_sidebar(img: np.ndarray) -> np.ndarray:
    """Append the controls panel to the right of the main display."""
    controls = draw_controls_panel(img.shape[0])
    return np.hstack([img, controls])
