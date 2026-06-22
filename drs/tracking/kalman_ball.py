"""Kalman filter ball tracker for gap filling between detections."""

from __future__ import annotations

import numpy as np


class BallTracker:
    """Simple constant-velocity Kalman filter for 2D ball position."""

    def __init__(self):
        self._state: np.ndarray | None = None  # [x, y, vx, vy]
        self._cov: np.ndarray | None = None
        self._frames_since_update = 0
        self._max_predict_frames = 10

    def _init_state(self, x: int, y: int) -> None:
        self._state = np.array([float(x), float(y), 0.0, 0.0])
        self._cov = np.eye(4) * 100
        self._frames_since_update = 0

    def update(self, x: int, y: int) -> None:
        if self._state is None:
            self._init_state(x, y)
            return

        z = np.array([float(x), float(y)])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.eye(2) * 25
        S = H @ self._cov @ H.T + R
        K = self._cov @ H.T @ np.linalg.inv(S)
        innovation = z - H @ self._state
        self._state = self._state + K @ innovation
        self._cov = (np.eye(4) - K @ H) @ self._cov
        self._frames_since_update = 0

    def predict(self) -> tuple[int, int] | None:
        if self._state is None:
            return None

        if self._frames_since_update >= self._max_predict_frames:
            return None

        F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        Q = np.eye(4) * 0.5
        self._state = F @ self._state
        self._cov = F @ self._cov @ F.T + Q
        self._frames_since_update += 1
        return int(self._state[0]), int(self._state[1])

    def reset(self) -> None:
        self._state = None
        self._cov = None
        self._frames_since_update = 0
