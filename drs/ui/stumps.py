"""Stump corridor drawing helpers."""

from __future__ import annotations

import statistics

import cv2
import numpy as np

from drs.config import DRSConfig
from drs.fusion.calibration import StumpPoints, corridor_bounds_at_y, stump_x_at_y


def _band_row_stats(mask: np.ndarray, y_start: int, y_end: int) -> tuple[int, int, int] | None:
    """Return median center x, pitch width, and row y for a horizontal band of the pitch mask."""
    centers: list[int] = []
    widths: list[int] = []
    rows: list[int] = []
    y_start = max(0, y_start)
    y_end = min(mask.shape[0] - 1, y_end)

    for y in range(y_start, y_end + 1):
        cols = np.where(mask[y] > 0)[0]
        if len(cols) < 2:
            continue
        widths.append(int(cols[-1] - cols[0]))
        centers.append(int((cols[0] + cols[-1]) // 2))
        rows.append(y)

    if not centers:
        return None

    mid = len(centers) // 2
    order = sorted(range(len(centers)), key=lambda i: centers[i])
    cx = centers[order[mid]]
    pw = int(statistics.median(widths))
    y_mid = rows[order[mid]]
    return cx, pw, y_mid


def stump_points_from_pitch_contour(
    pitch_contours: list,
    config: DRSConfig,
    frame_h: int,
    frame_w: int | None = None,
) -> StumpPoints | None:
    """Derive fixed perspective stump points from pitch mask row slices (not full bbox)."""
    if not pitch_contours:
        return None

    pitch_cnt = max(pitch_contours, key=cv2.contourArea)
    if cv2.contourArea(pitch_cnt) <= config.pitch_area_min:
        return None

    w = frame_w or pitch_cnt.max(axis=0)[0][0] + 1
    mask = np.zeros((frame_h, w), dtype=np.uint8)
    cv2.drawContours(mask, [pitch_cnt], -1, 255, -1)

    px, py, _bw, bh = cv2.boundingRect(pitch_cnt)
    band = max(8, bh // 5)

    bowler_stats = _band_row_stats(mask, py, py + band)
    striker_stats = _band_row_stats(mask, py + bh - band, py + bh - 1)
    if bowler_stats is None or striker_stats is None:
        return None

    bowler_cx, bowler_pw, bowler_y = bowler_stats
    striker_cx, striker_pw, striker_y = striker_stats

    half_bowler = max(4, int(bowler_pw * config.stump_width_ratio))
    half_striker = max(4, int(striker_pw * config.stump_width_ratio))

    return StumpPoints(
        striker_off=(striker_cx - half_striker, striker_y),
        striker_leg=(striker_cx + half_striker, striker_y),
        bowler_off=(bowler_cx - half_bowler, bowler_y),
        bowler_leg=(bowler_cx + half_bowler, bowler_y),
    )


def draw_stump_corridor(
    img: np.ndarray,
    stump_points: StumpPoints,
    *,
    line_color: tuple[int, int, int] = (0, 255, 255),
    line_thickness: int = 2,
    fill: bool = False,
    fill_alpha: float = 0.12,
) -> np.ndarray:
    """Draw off-off and leg-leg perspective stump lines across the full frame."""
    out = img.copy()
    h = img.shape[0]

    off_top = (int(stump_x_at_y(stump_points.striker_off, stump_points.bowler_off, 0)), 0)
    off_bot = (int(stump_x_at_y(stump_points.striker_off, stump_points.bowler_off, h - 1)), h - 1)
    leg_top = (int(stump_x_at_y(stump_points.striker_leg, stump_points.bowler_leg, 0)), 0)
    leg_bot = (int(stump_x_at_y(stump_points.striker_leg, stump_points.bowler_leg, h - 1)), h - 1)

    if fill:
        poly = np.array([
            [int(corridor_bounds_at_y(stump_points, 0)[0]), 0],
            [int(corridor_bounds_at_y(stump_points, 0)[1]), 0],
            [int(corridor_bounds_at_y(stump_points, h - 1)[1]), h - 1],
            [int(corridor_bounds_at_y(stump_points, h - 1)[0]), h - 1],
        ], dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], (0, 180, 180))
        out = cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0)

    cv2.line(out, off_top, off_bot, line_color, line_thickness)
    cv2.line(out, leg_top, leg_bot, line_color, line_thickness)
    return out


def score_pitch_frame_for_stump_lock(
    pitch_contours: list,
    config: DRSConfig,
    frame_h: int,
    frame_w: int,
) -> float:
    """Higher score = clearer striker-end pitch row for locking stumps."""
    if not pitch_contours:
        return -1.0
    pitch_cnt = max(pitch_contours, key=cv2.contourArea)
    area = cv2.contourArea(pitch_cnt)
    if area <= config.pitch_area_min:
        return -1.0

    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.drawContours(mask, [pitch_cnt], -1, 255, -1)
    _px, py, _bw, bh = cv2.boundingRect(pitch_cnt)
    band = max(8, bh // 5)
    striker_stats = _band_row_stats(mask, py + bh - band, py + bh - 1)
    if striker_stats is None:
        return -1.0
    _cx, striker_pw, _y = striker_stats
    return float(striker_pw) + area * 0.001
