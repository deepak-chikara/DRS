"""Mutable runtime state for DRS pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BallState:
    x: int = 0
    y: int = 0
    x_prev: int = 0
    y_prev: int = 0
    prev_x_diff: int = 0
    prev_y_diff: int = 0
    source: str = "none"  # yolo | color | tracker | fused
    confidence: float = 0.0


@dataclass
class DRSState:
    pitch_point: tuple[int, int] | None = None
    impact_point: tuple[int, int] | None = None
    impact_locked: bool = False
    pitch_counter: int = 0
    lbw_detected: bool = False
    pad_detected: bool = False
    last_motion_class: str = "Motion"
    bat_leg: int = 10000
    ball: BallState = field(default_factory=BallState)
    fused_ball_pitch: tuple[float, float] | None = None
    sync_quality_ms: float | None = None
    delivery_count: int = 0
    trajectory_points: list[tuple[float, float]] = field(default_factory=list)
    trajectory_pitch_points: list[tuple[float, float]] = field(default_factory=list)
    verdict: str = ""
    verdict_reason: str = ""

    def update_ball_prev(self) -> None:
        b = self.ball
        b.x_prev = b.x
        b.y_prev = b.y

    def update_ball_diff(self) -> None:
        b = self.ball
        if b.x_prev != 0 or b.y_prev != 0:
            b.prev_x_diff = b.x - b.x_prev
            b.prev_y_diff = b.y - b.y_prev

    def reset_delivery(self) -> None:
        """Reset per-delivery flags for a new ball or video restart."""
        self.pitch_point = None
        self.impact_point = None
        self.impact_locked = False
        self.pitch_counter = 0
        self.lbw_detected = False
        self.pad_detected = False
        self.last_motion_class = "Motion"
        self.bat_leg = 10000
        self.verdict = ""
        self.verdict_reason = ""
        self.trajectory_points.clear()
        self.trajectory_pitch_points.clear()
