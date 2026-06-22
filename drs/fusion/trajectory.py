"""Ball trajectory estimation on pitch plane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryResult:
    points: list[tuple[float, float]]
    predicted_impact: tuple[float, float] | None
    predicted_pitch: tuple[float, float] | None


class TrajectoryEstimator:
    """Fit parabolic path and extrapolate to key lines."""

    def __init__(self, min_points: int = 5):
        self.min_points = min_points
        self._history: list[tuple[float, float]] = []

    def add_point(self, pitch_x: float, pitch_y: float) -> None:
        self._history.append((pitch_x, pitch_y))
        if len(self._history) > 60:
            self._history = self._history[-60:]

    def reset(self) -> None:
        self._history.clear()

    def estimate(self, batsman_line_y: float = 0.3, stump_line_y: float = 0.0) -> TrajectoryResult:
        if len(self._history) < self.min_points:
            return TrajectoryResult(self._history.copy(), None, None)

        xs = np.array([p[0] for p in self._history])
        ys = np.array([p[1] for p in self._history])
        t = np.arange(len(xs))

        try:
            x_coeff = np.polyfit(t, xs, 2)
            y_coeff = np.polyfit(t, ys, 2)
        except (np.linalg.LinAlgError, ValueError):
            return TrajectoryResult(self._history.copy(), None, None)

        def eval_at_y(target_y: float) -> tuple[float, float] | None:
            y_poly = np.poly1d(y_coeff)
            roots = (y_poly - target_y).roots
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real >= t[-1]]
            if not real_roots:
                return None
            t_hit = max(real_roots)
            x_hit = float(np.polyval(x_coeff, t_hit))
            return x_hit, target_y

        return TrajectoryResult(
            points=self._history.copy(),
            predicted_impact=eval_at_y(batsman_line_y),
            predicted_pitch=eval_at_y(stump_line_y),
        )
