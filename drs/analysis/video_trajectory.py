"""Extract ball trajectory from a recorded video for pitch diagram replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from drs.config import DRSConfig, load_config
from drs.decision.events import EventDetector
from drs.detectors.hybrid import HybridDetector
from drs.engine import DRSEngine
from drs.fusion.calibration import StumpPoints, pixel_to_pitch_normalized
from drs.state import DRSState


@dataclass
class DeliveryTrajectory:
    delivery_index: int
    pitch_points: list[tuple[float, float]]
    pixel_points: list[tuple[float, float]]
    frame_h: int = 720
    pitch_bounce: tuple[float, float] | None = None
    impact: tuple[float, float] | None = None
    verdict: str = ""
    verdict_reason: str = ""
    stump_points: StumpPoints | None = None


def _to_pitch_point(
    x: int,
    y: int,
    frame_w: int,
    frame_h: int,
    homography: np.ndarray | None,
    stump_points: StumpPoints | None,
) -> tuple[float, float]:
    return pixel_to_pitch_normalized(
        x, y,
        frame_w=frame_w,
        frame_h=frame_h,
        homography=homography,
        stump_points=stump_points,
    )


def extract_trajectories_from_video(
    video_path: str | Path,
    config: DRSConfig | None = None,
) -> list[DeliveryTrajectory]:
    """Process every frame and collect per-delivery pitch-plane trajectories."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    cfg = config or load_config(Path(__file__).resolve().parent.parent / "config" / "default.yaml")
    cfg.video_path = str(path.resolve())
    cfg.mode = "file"

    engine = DRSEngine(cfg)
    ok, err = engine.open()
    if not ok:
        raise RuntimeError(err)

    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        engine.close()
        raise RuntimeError(f"Cannot open video: {path}")

    hsv = cfg.ball_hsv or {
        "hmin": 10, "smin": 44, "vmin": 192,
        "hmax": 125, "smax": 114, "vmax": 255,
    }
    detector = HybridDetector(
        detection_mode=cfg.detection_mode,
        hsv_values=hsv,
        rgb_lower=np.array(cfg.batsman_rgb_lower),
        rgb_upper=np.array(cfg.batsman_rgb_upper),
        canny1=cfg.batsman_canny1,
        canny2=cfg.batsman_canny2,
        detection_scale=cfg.detection_scale,
        pitch_cache_frames=1,
    )
    events = EventDetector(cfg)
    state = DRSState()
    homography = engine._pitch_homography

    deliveries: list[DeliveryTrajectory] = []
    current_pitch: list[tuple[float, float]] = []
    current_pixel: list[tuple[float, float]] = []
    delivery_idx = 0
    prev_frame = -1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    last_fw, last_fh = 1280, 720
    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        last_fh, last_fw = frame.shape[:2]

        if prev_frame >= 0 and (frame_i < prev_frame or frame_i - prev_frame > 1):
            if current_pitch:
                deliveries.append(DeliveryTrajectory(
                    delivery_index=delivery_idx,
                    pitch_points=current_pitch.copy(),
                    pixel_points=current_pixel.copy(),
                    frame_h=last_fh,
                    pitch_bounce=_bounce_from_state(
                        state, last_fw, last_fh, homography, engine.stump_points,
                    ),
                    impact=_impact_from_state(
                        state, last_fw, last_fh, homography, engine.stump_points,
                    ),
                    verdict=state.verdict,
                    verdict_reason=state.verdict_reason,
                    stump_points=engine.stump_points,
                ))
            state.reset_delivery()
            events.reset()
            current_pitch.clear()
            current_pixel.clear()
            delivery_idx += 1

        prev_frame = frame_i
        fh, fw = frame.shape[:2]

        state.update_ball_prev()
        result = detector.process(frame)
        engine._try_lock_stumps_from_pitch(result.pitch_contours, fh, fw)
        state.ball.x = result.ball_x
        state.ball.y = result.ball_y
        state.update_ball_diff()
        events.process_frame(state, result.pitch_contours, result.batsman_contours)

        if result.ball_x != 0 or result.ball_y != 0:
            if events._delivery_in_progress(state):
                pt = _to_pitch_point(
                    result.ball_x, result.ball_y, fw, fh, homography, engine.stump_points,
                )
                if not current_pitch or current_pitch[-1] != pt:
                    current_pitch.append(pt)
                    current_pixel.append((float(result.ball_x), float(result.ball_y)))

        frame_i += 1

    if current_pitch:
        deliveries.append(DeliveryTrajectory(
            delivery_index=delivery_idx,
            pitch_points=current_pitch,
            pixel_points=current_pixel,
            frame_h=last_fh,
            pitch_bounce=_bounce_from_state(state, last_fw, last_fh, homography, engine.stump_points),
            impact=_impact_from_state(state, last_fw, last_fh, homography, engine.stump_points),
            verdict=state.verdict,
            verdict_reason=state.verdict_reason,
            stump_points=engine.stump_points,
        ))

    cap.release()
    saved_stumps = engine.stump_points
    engine.close()

    if not deliveries and total > 0:
        deliveries = _fallback_single_delivery(path, cfg, homography, saved_stumps)

    return deliveries


def _bounce_from_state(
    state: DRSState,
    fw: int,
    fh: int,
    homography: np.ndarray | None,
    stump_points: StumpPoints | None,
) -> tuple[float, float] | None:
    if state.pitch_point is None:
        return None
    x, y = state.pitch_point
    return _to_pitch_point(x, y, fw, fh, homography, stump_points)


def _impact_from_state(
    state: DRSState,
    fw: int,
    fh: int,
    homography: np.ndarray | None,
    stump_points: StumpPoints | None,
) -> tuple[float, float] | None:
    if state.impact_point is None:
        return None
    x, y = state.impact_point
    return _to_pitch_point(x, y, fw, fh, homography, stump_points)


def _fallback_single_delivery(
    path: Path,
    cfg: DRSConfig,
    homography: np.ndarray | None,
    stump_points: StumpPoints | None,
) -> list[DeliveryTrajectory]:
    """Use full-video ball detections when delivery boundaries are unclear."""
    cap = cv2.VideoCapture(str(path))
    hsv = cfg.ball_hsv or {"hmin": 10, "smin": 44, "vmin": 192, "hmax": 125, "smax": 114, "vmax": 255}
    detector = HybridDetector(
        detection_mode=cfg.detection_mode,
        hsv_values=hsv,
        rgb_lower=np.array(cfg.batsman_rgb_lower),
        rgb_upper=np.array(cfg.batsman_rgb_upper),
        canny1=cfg.batsman_canny1,
        canny2=cfg.batsman_canny2,
        detection_scale=cfg.detection_scale,
        pitch_cache_frames=1,
    )
    pitch_pts: list[tuple[float, float]] = []
    pixel_pts: list[tuple[float, float]] = []
    frame_h = 720
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_h = frame.shape[0]
        r = detector.process(frame)
        if r.ball_x != 0 or r.ball_y != 0:
            pt = _to_pitch_point(
                r.ball_x, r.ball_y, frame.shape[1], frame.shape[0], homography, stump_points,
            )
            if not pitch_pts or pitch_pts[-1] != pt:
                pitch_pts.append(pt)
                pixel_pts.append((float(r.ball_x), float(r.ball_y)))
    cap.release()
    if not pitch_pts:
        return []
    return [DeliveryTrajectory(
        delivery_index=0,
        pitch_points=pitch_pts,
        pixel_points=pixel_pts,
        frame_h=frame_h,
        stump_points=stump_points,
    )]
