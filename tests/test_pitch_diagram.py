"""Pitch diagram rendering tests."""

import tempfile
from pathlib import Path

import numpy as np

from drs.fusion.calibration import StumpPoints
from drs.ui.pitch_diagram import (
    build_side_profile,
    export_combined_diagram_video,
    export_diagram_video,
    refine_ball_track,
    render_combined_diagram,
    render_pitch_diagram,
    render_side_diagram,
    tracking_quality,
    tracking_verdict_label,
)


def test_render_pitch_diagram_no_crash():
    pts = [(0.5, 0.9), (0.48, 0.7), (0.46, 0.5), (0.45, 0.3)]
    img = render_pitch_diagram(pts, progress=2)
    assert img.shape[0] > 0
    assert img.shape[1] > 0


def test_render_pitch_diagram_draws_corridor_lines():
    pts = [(0.5, 0.9), (0.5, 0.5), (0.5, 0.2)]
    img = render_pitch_diagram(pts, progress=2)
    yellow = np.all(img == (0, 200, 200), axis=-1)
    assert yellow.sum() > 20


def test_refine_ball_track_filters_noise():
    good = [(0.42, 0.9), (0.43, 0.7), (0.44, 0.5), (0.45, 0.3), (0.46, 0.1)]
    noisy = good + [(0.8, 0.55), (0.2, 0.52), (0.75, 0.48)]
    refined, _ = refine_ball_track(noisy)
    assert len(refined) >= len(good)
    for i in range(1, len(refined)):
        assert refined[i][1] <= refined[i - 1][1] + 0.01


def test_build_side_profile():
    pitch = [(0.5, 0.9), (0.5, 0.5), (0.5, 0.2)]
    pixels = [(640.0, 600.0), (640.0, 400.0), (640.0, 200.0)]
    profile = build_side_profile(pitch, pixels, frame_h=720)
    assert len(profile) >= 3
    assert profile[0][1] < profile[-1][1]


def test_build_side_profile_with_stumps_uses_pitch_ground():
    stumps = StumpPoints(
        striker_off=(615, 221),
        striker_leg=(640, 221),
        bowler_off=(611, 495),
        bowler_leg=(643, 492),
    )
    pitch_ground = [(0.5, 0.5)]
    pitch_air = [(0.5, 0.5)]
    ground_y = 358.0
    on_ground = build_side_profile(
        pitch_ground, [(627.0, ground_y)], frame_h=720, stump_points=stumps,
    )
    airborne = build_side_profile(
        pitch_air, [(627.0, ground_y - 50.0)], frame_h=720, stump_points=stumps,
    )
    assert on_ground[0][1] < 0.05
    assert airborne[0][1] > on_ground[0][1]


def test_build_side_profile_degenerate_lengths_no_crash():
    """Live playback can produce identical pitch lengths — must not crash polyfit."""
    pitch = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
    pixels = [(640.0, 400.0), (640.0, 380.0), (640.0, 360.0), (640.0, 340.0)]
    profile = build_side_profile(pitch, pixels, frame_h=720)
    assert len(profile) >= 1
    combined = render_combined_diagram(pitch, pixels, progress=2, frame_h=720, live_ball=(0.5, 0.5))
    assert combined.shape[0] > 0


def test_render_side_diagram_no_crash():
    pitch = [(0.5, 0.9), (0.48, 0.7), (0.46, 0.5), (0.45, 0.3)]
    pixels = [(640.0, 650.0), (640.0, 500.0), (640.0, 350.0), (640.0, 200.0)]
    profile = build_side_profile(pitch, pixels, frame_h=720)
    img = render_side_diagram(profile, progress=2, source_point_count=4)
    assert img.shape[0] > 0
    assert img.shape[1] > 0


def test_render_combined_diagram_width():
    pitch = [(0.5, 0.9), (0.5, 0.5), (0.5, 0.2)]
    pixels = [(640.0, 600.0), (640.0, 400.0), (640.0, 200.0)]
    single = render_pitch_diagram(pitch, progress=2)
    combined = render_combined_diagram(pitch, pixels, progress=2, frame_h=720)
    assert combined.shape[1] >= single.shape[1] * 2 - 4


def test_render_pitch_diagram_live_ball():
    img = render_pitch_diagram([], live_ball=(0.5, 0.5))
    assert img.shape[0] > 0
    img2 = render_pitch_diagram([(0.5, 0.9)], live_ball=(0.5, 0.5))
    assert img2.shape[0] > 0


def test_tracking_quality_review():
    assert tracking_quality(3) == "review"
    assert "REVIEW" in tracking_verdict_label(3)


def test_tracking_quality_insufficient():
    assert tracking_quality(1) == "insufficient"
    assert "Insufficient" in tracking_verdict_label(1)


def test_export_diagram_video():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diag.mp4"
        export_diagram_video([(0.5, 0.9), (0.5, 0.5), (0.5, 0.2)], str(out), fps=10)
        assert out.is_file()


def test_export_combined_diagram_video():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "combined.mp4"
        pitch = [(0.5, 0.9), (0.5, 0.5), (0.5, 0.2)]
        pixels = [(640.0, 600.0), (640.0, 400.0), (640.0, 200.0)]
        export_combined_diagram_video(pitch, pixels, str(out), fps=10, frame_h=720)
        assert out.is_file()
        sample = render_combined_diagram(pitch, pixels, progress=0, frame_h=720)
        assert sample.shape[1] >= 600
