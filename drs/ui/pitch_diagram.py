"""Top-down pitch diagram and side elevation for animated ball trajectory replay."""

from __future__ import annotations

import cv2
import numpy as np

PITCH_LENGTH_M = 20.12
PITCH_WIDTH_M = 3.05
STUMP_HEIGHT_M = 0.711
MAX_DISPLAY_HEIGHT_M = 3.0
# Three stumps span 9 inches (22.86 cm) across a 3.05 m pitch.
STUMP_SET_WIDTH_FRAC = 22.86 / (PITCH_WIDTH_M * 100)
STUMP_END_DEPTH_FRAC = 0.022
STUMP_PX_SPAN_FRAC = 0.12


def tracking_quality(point_count: int) -> str:
    if point_count >= 5:
        return "good"
    if point_count >= 2:
        return "review"
    return "insufficient"


def tracking_verdict_label(point_count: int) -> str:
    q = tracking_quality(point_count)
    if q == "good":
        return ""
    if q == "review":
        return "REVIEW — sparse tracking"
    return "Insufficient tracking — use recorded clip"


def _best_monotonic_chain(
    points: list[tuple[float, float]],
    *,
    decreasing: bool = True,
    max_lateral: float = 0.12,
    max_step: float = 0.22,
) -> list[tuple[float, float]]:
    """Keep the longest plausible delivery path (bowler → striker)."""
    if len(points) <= 2:
        return list(points)

    best: list[tuple[float, float]] = []
    for start in range(len(points)):
        chain = [points[start]]
        for idx in range(start + 1, len(points)):
            point = points[idx]
            prev = chain[-1]
            d_len = point[1] - prev[1]
            if decreasing and d_len > 0.035:
                continue
            if not decreasing and d_len < -0.035:
                continue
            if abs(d_len) > max_step:
                continue
            if abs(point[0] - prev[0]) > max_lateral:
                continue
            chain.append(point)
        if len(chain) > len(best):
            best = chain
    return best if len(best) >= 2 else list(points[:2])


def _smooth_resample_track(
    pitch: list[tuple[float, float]],
    pixel: list[tuple[float, float]] | None,
    sample_count: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]] | None]:
    if len(pitch) < 2:
        return list(pitch), pixel

    t = np.linspace(0.0, 1.0, len(pitch))
    t_new = np.linspace(0.0, 1.0, sample_count)
    deg = min(2, len(pitch) - 1)
    nx = np.array([p[0] for p in pitch], dtype=float)
    ny = np.array([p[1] for p in pitch], dtype=float)
    cx = np.polyfit(t, nx, deg)
    cy = np.polyfit(t, ny, deg)
    new_pitch = [
        (float(np.clip(np.polyval(cx, ti), 0.0, 1.0)), float(np.clip(np.polyval(cy, ti), 0.0, 1.0)))
        for ti in t_new
    ]

    new_pixel: list[tuple[float, float]] | None = None
    if pixel and len(pixel) == len(pitch):
        px = np.array([p[0] for p in pixel], dtype=float)
        py = np.array([p[1] for p in pixel], dtype=float)
        cpx = np.polyfit(t, px, deg)
        cpy = np.polyfit(t, py, deg)
        new_pixel = [(float(np.polyval(cpx, ti)), float(np.polyval(cpy, ti))) for ti in t_new]
    return new_pitch, new_pixel


def refine_ball_track(
    pitch_points: list[tuple[float, float]],
    pixel_points: list[tuple[float, float]] | None = None,
    *,
    sample_count: int = 32,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]] | None]:
    """Filter noisy detections and return a smooth bowler→striker track."""
    if not pitch_points:
        return [], pixel_points

    pixels = list(pixel_points or [])
    if pixels and len(pixels) != len(pitch_points):
        count = min(len(pixels), len(pitch_points))
        pitch_points = pitch_points[:count]
        pixels = pixels[:count]

    decreasing = pitch_points[0][1] >= pitch_points[-1][1]
    order = sorted(range(len(pitch_points)), key=lambda i: pitch_points[i][1], reverse=decreasing)
    ordered_pitch = [pitch_points[i] for i in order]
    ordered_pixel = [pixels[i] for i in order] if len(pixels) == len(pitch_points) else []

    chain = _best_monotonic_chain(ordered_pitch, decreasing=decreasing)
    chain_pixels: list[tuple[float, float]] = []
    if ordered_pixel:
        for point in chain:
            nearest = min(
                range(len(ordered_pitch)),
                key=lambda i: (ordered_pitch[i][0] - point[0]) ** 2 + (ordered_pitch[i][1] - point[1]) ** 2,
            )
            chain_pixels.append(ordered_pixel[nearest])

    return _smooth_resample_track(
        chain,
        chain_pixels if chain_pixels else None,
        max(sample_count, len(chain) * 2),
    )


def _append_live_point(
    pitch_points: list[tuple[float, float]],
    pixel_points: list[tuple[float, float]] | None,
    live_ball: tuple[float, float] | None,
    live_pixel: tuple[float, float] | None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]] | None]:
    pts = list(pitch_points)
    pixels = list(pixel_points or [])
    if live_ball is None:
        return pts, pixels if pixels else None
    if not pts or abs(live_ball[0] - pts[-1][0]) > 0.002 or abs(live_ball[1] - pts[-1][1]) > 0.002:
        pts.append(live_ball)
        if live_pixel is not None:
            pixels.append(live_pixel)
    else:
        pts[-1] = live_ball
        if live_pixel is not None and pixels:
            pixels[-1] = live_pixel
        elif live_pixel is not None:
            pixels.append(live_pixel)
    return pts, pixels if pixels else None


def _draw_progress_count(progress: int | None, raw_count: int, path_len: int) -> int:
    if path_len <= 0:
        return 0
    if progress is None:
        return path_len
    if raw_count <= 1:
        return min(progress + 1, path_len)
    return min(path_len, max(1, round((progress + 1) * path_len / raw_count)))


def _stump_half_width_px(pitch_w: int) -> int:
    return max(3, int(pitch_w * STUMP_SET_WIDTH_FRAC / 2))


def _draw_stump_end(
    img: np.ndarray,
    cx: int,
    y_edge: int,
    pitch_w: int,
    *,
    at_striker: bool,
) -> tuple[int, int]:
    """Draw stump block with yellow off/leg marks; returns (off_x, leg_x)."""
    half = _stump_half_width_px(pitch_w)
    depth = max(6, int(pitch_w * STUMP_END_DEPTH_FRAC))
    off_x = cx - half
    leg_x = cx + half

    if at_striker:
        y1, y2 = y_edge, y_edge + depth
    else:
        y1, y2 = y_edge - depth, y_edge

    cv2.rectangle(img, (off_x, y1), (leg_x, y2), (140, 140, 140), -1)
    cv2.rectangle(img, (off_x, y1), (leg_x, y2), (220, 220, 220), 1)

    for frac in (-0.55, 0.0, 0.55):
        sx = int(cx + half * frac)
        cv2.line(img, (sx, y1 + 1), (sx, y2 - 1), (250, 250, 250), 2)

    cv2.line(img, (off_x, y1), (off_x, y2), (0, 255, 255), 2)
    cv2.line(img, (leg_x, y1), (leg_x, y2), (0, 255, 255), 2)

    return off_x, leg_x


def render_pitch_diagram(
    pitch_points: list[tuple[float, float]],
    *,
    width: int = 320,
    height: int = 480,
    progress: int | None = None,
    pitch_bounce: tuple[float, float] | None = None,
    impact: tuple[float, float] | None = None,
    live_ball: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render Hawk-Eye-style top-down pitch view with stumps at each end."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 90, 30)

    margin = 24
    pitch_w = width - 2 * margin
    pitch_h = height - 2 * margin
    x0, y0 = margin, margin
    x1, y1 = x0 + pitch_w, y0 + pitch_h
    cx = x0 + pitch_w // 2

    cv2.rectangle(img, (x0, y0), (x1, y1), (40, 120, 40), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)

    off_striker, leg_striker = _draw_stump_end(img, cx, y0, pitch_w, at_striker=True)
    off_bowler, leg_bowler = _draw_stump_end(img, cx, y1, pitch_w, at_striker=False)

    cv2.line(img, (off_striker, y0), (off_bowler, y1), (0, 200, 200), 1)
    cv2.line(img, (leg_striker, y0), (leg_bowler, y1), (0, 200, 200), 1)

    def to_px(nx: float, ny: float) -> tuple[int, int]:
        """Striker end at top, bowler end at bottom."""
        px = int(x0 + nx * pitch_w)
        py = int(y0 + ny * pitch_h)
        return px, py

    display_pts, _ = _append_live_point(pitch_points, None, live_ball, None)
    path, _ = refine_ball_track(display_pts)
    raw_count = len(display_pts)
    if live_ball is not None:
        draw_count = len(path)
    else:
        draw_count = _draw_progress_count(progress, raw_count, len(path))
    if draw_count >= 2:
        pts = np.array([to_px(p[0], p[1]) for p in path[:draw_count]], dtype=np.int32)
        dashed = tracking_quality(raw_count) == "review"
        if dashed:
            for i in range(len(pts) - 1):
                cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 255), 2)
        else:
            cv2.polylines(img, [pts], False, (0, 0, 255), 2)

    if draw_count > 0 and path:
        last = path[min(draw_count - 1, len(path) - 1)]
        bx, by = to_px(last[0], last[1])
        cv2.circle(img, (bx, by), 6, (0, 0, 255), -1)
    if live_ball is not None:
        lx, ly = to_px(live_ball[0], live_ball[1])
        cv2.circle(img, (lx, ly), 8, (0, 255, 255), -1)
        cv2.circle(img, (lx, ly), 8, (0, 0, 255), 2)

    if pitch_bounce is not None:
        cv2.circle(img, to_px(pitch_bounce[0], pitch_bounce[1]), 5, (255, 0, 0), -1)
    if impact is not None:
        cv2.circle(img, to_px(impact[0], impact[1]), 6, (0, 0, 255), 2)

    label = tracking_verdict_label(raw_count)
    if label:
        cv2.putText(img, label[:42], (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

    cv2.putText(img, "Striker", (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    cv2.putText(img, "Bowler", (x0 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    return img


def build_side_profile(
    pitch_points: list[tuple[float, float]],
    pixel_points: list[tuple[float, float]],
    frame_h: int,
) -> list[tuple[float, float]]:
    """Build (length_norm, height_m) pairs from paired pitch and pixel trajectories."""
    refined_pitch, refined_pixel = refine_ball_track(pitch_points, pixel_points, sample_count=24)
    if not refined_pitch:
        return []

    if not refined_pixel:
        return [(ny, 0.0) for _, ny in refined_pitch]

    count = min(len(refined_pitch), len(refined_pixel))
    py_values = [refined_pixel[i][1] for i in range(count)]
    ground_y = max(py_values)
    if ground_y <= 0:
        ground_y = frame_h * 0.88

    stump_px_span = max(frame_h * STUMP_PX_SPAN_FRAC, 1.0)
    m_per_px = STUMP_HEIGHT_M / stump_px_span

    profile: list[tuple[float, float]] = []
    for i in range(count):
        _, ny = refined_pitch[i]
        _, py = refined_pixel[i]
        height_m = max(0.0, (ground_y - py) * m_per_px)
        profile.append((ny, min(height_m, MAX_DISPLAY_HEIGHT_M)))

    if len(profile) >= 3:
        lengths = np.array([p[0] for p in profile], dtype=float)
        heights = np.array([p[1] for p in profile], dtype=float)
        deg = min(2, len(profile) - 1)
        coeff = np.polyfit(lengths, heights, deg)
        profile = [
            (float(lengths[i]), float(max(0.0, min(np.polyval(coeff, lengths[i]), MAX_DISPLAY_HEIGHT_M))))
            for i in range(len(profile))
        ]

    return profile


def _side_event_height(
    pitch_event: tuple[float, float] | None,
    side_profile: list[tuple[float, float]],
) -> tuple[float, float] | None:
    if pitch_event is None or not side_profile:
        return None
    _, event_ny = pitch_event
    best = min(side_profile, key=lambda p: abs(p[0] - event_ny))
    return best[0], best[1]


def render_side_diagram(
    side_profile: list[tuple[float, float]],
    *,
    width: int = 320,
    height: int = 480,
    progress: int | None = None,
    pitch_bounce: tuple[float, float] | None = None,
    impact: tuple[float, float] | None = None,
    source_point_count: int | None = None,
    live_active: bool = False,
) -> np.ndarray:
    """Render side elevation (length vs height) with stumps at each end."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 50, 30)

    margin = 24
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin - 16
    x0, y0 = margin, margin
    x1, y1 = x0 + plot_w, y0 + plot_h
    ground_y = y1

    cv2.rectangle(img, (x0, ground_y - 4), (x1, y1), (40, 100, 40), -1)
    cv2.line(img, (x0, ground_y), (x1, ground_y), (200, 200, 200), 1)

    def length_to_x(length_norm: float) -> int:
        return int(x0 + length_norm * plot_w)

    def height_to_y(height_m: float) -> int:
        frac = min(height_m / MAX_DISPLAY_HEIGHT_M, 1.0)
        return int(ground_y - frac * plot_h)

    stump_w = max(8, plot_w // 24)
    stump_h_px = int((STUMP_HEIGHT_M / MAX_DISPLAY_HEIGHT_M) * plot_h)

    for length_norm in (0.0, 1.0):
        sx = length_to_x(length_norm)
        sy_base = ground_y
        sy_top = ground_y - stump_h_px
        x_left = sx - stump_w // 2
        x_right = sx + stump_w // 2
        cv2.rectangle(img, (x_left, sy_top), (x_right, sy_base), (140, 140, 140), -1)
        cv2.rectangle(img, (x_left, sy_top), (x_right, sy_base), (220, 220, 220), 1)
        cv2.line(img, (x_left, sy_top), (x_right, sy_top), (0, 255, 255), 1)

    path = list(side_profile)
    raw_count = source_point_count if source_point_count is not None else len(side_profile)
    if live_active:
        draw_count = len(path)
    else:
        draw_count = _draw_progress_count(progress, raw_count, len(path))
    if draw_count >= 2:
        pts = np.array(
            [(length_to_x(p[0]), height_to_y(p[1])) for p in path[:draw_count]],
            dtype=np.int32,
        )
        dashed = tracking_quality(raw_count) == "review"
        if dashed:
            for i in range(len(pts) - 1):
                cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 255), 2)
        else:
            cv2.polylines(img, [pts], False, (0, 0, 255), 2)

    if draw_count > 0 and path:
        last = path[min(draw_count - 1, len(path) - 1)]
        cv2.circle(img, (length_to_x(last[0]), height_to_y(last[1])), 6, (0, 0, 255), -1)

    bounce_side = _side_event_height(pitch_bounce, side_profile)
    if bounce_side is not None:
        cv2.circle(
            img,
            (length_to_x(bounce_side[0]), height_to_y(bounce_side[1])),
            5,
            (255, 0, 0),
            -1,
        )

    impact_side = _side_event_height(impact, side_profile)
    if impact_side is not None:
        cv2.circle(
            img,
            (length_to_x(impact_side[0]), height_to_y(impact_side[1])),
            6,
            (0, 0, 255),
            2,
        )

    label = tracking_verdict_label(raw_count)
    if label:
        cv2.putText(img, label[:42], (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

    cv2.putText(img, "Striker", (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    cv2.putText(img, "Bowler", (x0 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    cv2.putText(img, "Side", (x0 + plot_w - 40, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(img, f"{STUMP_HEIGHT_M:.2f}m", (x0 + 4, height_to_y(STUMP_HEIGHT_M) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 200), 1)
    return img


def render_combined_diagram(
    pitch_points: list[tuple[float, float]],
    pixel_points: list[tuple[float, float]],
    *,
    progress: int | None = None,
    frame_h: int = 720,
    pitch_bounce: tuple[float, float] | None = None,
    impact: tuple[float, float] | None = None,
    panel_width: int = 320,
    panel_height: int = 480,
    live_ball: tuple[float, float] | None = None,
    live_pixel: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render top-down and side elevation panels side by side."""
    display_pts, display_px = _append_live_point(pitch_points, pixel_points, live_ball, live_pixel)
    live_active = live_ball is not None
    front = render_pitch_diagram(
        pitch_points,
        width=panel_width,
        height=panel_height,
        progress=progress,
        pitch_bounce=pitch_bounce,
        impact=impact,
        live_ball=live_ball,
    )
    side_profile = build_side_profile(
        display_pts,
        display_px or [],
        frame_h,
    )
    side = render_side_diagram(
        side_profile,
        width=panel_width,
        height=panel_height,
        progress=progress,
        pitch_bounce=pitch_bounce,
        impact=impact,
        source_point_count=len(display_pts),
        live_active=live_active,
    )
    if front.shape[0] != side.shape[0]:
        side = cv2.resize(side, (side.shape[1], front.shape[0]))
    return np.hstack([front, side])


def export_diagram_video(
    pitch_points: list[tuple[float, float]],
    output_path: str,
    fps: float = 30.0,
    *,
    pitch_bounce: tuple[float, float] | None = None,
    impact: tuple[float, float] | None = None,
) -> None:
    if not pitch_points:
        pitch_points = [(0.5, 0.02), (0.5, 0.5)]
    sample = render_pitch_diagram(pitch_points, progress=0)
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(len(pitch_points)):
        writer.write(render_pitch_diagram(
            pitch_points, progress=i, pitch_bounce=pitch_bounce, impact=impact,
        ))
    writer.release()


def export_combined_diagram_video(
    pitch_points: list[tuple[float, float]],
    pixel_points: list[tuple[float, float]],
    output_path: str,
    fps: float = 30.0,
    *,
    frame_h: int = 720,
    pitch_bounce: tuple[float, float] | None = None,
    impact: tuple[float, float] | None = None,
) -> None:
    if not pitch_points:
        pitch_points = [(0.5, 0.02), (0.5, 0.5)]
    sample = render_combined_diagram(
        pitch_points, pixel_points, progress=0, frame_h=frame_h,
        pitch_bounce=pitch_bounce, impact=impact,
    )
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(len(pitch_points)):
        writer.write(render_combined_diagram(
            pitch_points, pixel_points, progress=i, frame_h=frame_h,
            pitch_bounce=pitch_bounce, impact=impact,
        ))
    writer.release()
