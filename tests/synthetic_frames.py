"""Synthetic frames for CI when no sample video is committed."""

from __future__ import annotations

import cv2
import numpy as np


def make_synthetic_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """Green pitch, yellow stump lines, red ball — stable for regression tests."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (32, 90, 48)  # BGR turf
    cx = width // 2
    frame[height // 4 : 3 * height // 4, cx - 40 : cx + 40] = (40, 110, 55)
    cv2.line(frame, (cx - 50, height // 3), (cx - 50, 2 * height // 3), (80, 200, 220), 2)
    cv2.line(frame, (cx + 50, height // 3), (cx + 50, 2 * height // 3), (80, 200, 220), 2)
    cv2.circle(frame, (cx - 10, height // 2), 8, (40, 40, 220), -1)
    return frame
