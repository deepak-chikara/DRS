"""Hybrid detector: YOLO primary, color-based fallback."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from cvzone.ColorModule import ColorFinder

from ball_detect import ball_detect
from batsman import batsman_detect
from drs.detectors.yolo_detector import Detection, YOLODetector
from drs.tracking.kalman_ball import BallTracker
from pitch import pitch


@dataclass
class HybridResult:
    ball_x: int
    ball_y: int
    ball_source: str
    ball_confidence: float
    ball_overlay: np.ndarray | None
    batsman_contours: list
    yolo_person: Detection | None
    pitch_contours: list


def _scale_contours(contours: list, sx: float, sy: float) -> list:
    if sx == 1.0 and sy == 1.0:
        return contours
    scaled = []
    for cnt in contours:
        c = cnt.copy().astype(np.float64)
        c[:, :, 0] *= sx
        c[:, :, 1] *= sy
        scaled.append(c.astype(np.int32))
    return scaled


class HybridDetector:
    def __init__(
        self,
        detection_mode: str,
        hsv_values: dict,
        rgb_lower: np.ndarray,
        rgb_upper: np.ndarray,
        canny1: int,
        canny2: int,
        yolo_model: str = "yolov8n.pt",
        yolo_ball_conf: float = 0.25,
        yolo_person_conf: float = 0.4,
        detection_scale: float = 1.0,
        pitch_cache_frames: int = 15,
    ):
        self.detection_mode = detection_mode
        self.hsv_values = hsv_values
        self.rgb_lower = rgb_lower
        self.rgb_upper = rgb_upper
        self.canny1 = canny1
        self.canny2 = canny2
        self.detection_scale = max(0.25, min(1.0, detection_scale))
        self.pitch_cache_frames = max(1, pitch_cache_frames)
        self.color_finder = ColorFinder(False)
        self.yolo = YOLODetector(yolo_model, yolo_ball_conf, yolo_person_conf) if detection_mode != "color" else None
        self.tracker = BallTracker()
        self._pitch_contours: list = []
        self._pitch_cache_counter = 0

    def _resize_for_detection(self, img: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = img.shape[:2]
        scale = self.detection_scale
        if scale >= 1.0:
            return img, 1.0, 1.0
        sw = max(320, int(w * scale))
        sh = max(240, int(h * scale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        return small, w / sw, h / sh

    def process(self, img: np.ndarray) -> HybridResult:
        small, sx, sy = self._resize_for_detection(img)

        _, color_x, color_y = ball_detect(small, self.color_finder, self.hsv_values)
        if color_x:
            color_x = int(color_x * sx)
            color_y = int(color_y * sy)

        batsman_contours = _scale_contours(
            batsman_detect(small, self.rgb_lower, self.rgb_upper, self.canny1, self.canny2),
            sx,
            sy,
        )

        self._pitch_cache_counter += 1
        if self._pitch_cache_counter >= self.pitch_cache_frames or not self._pitch_contours:
            self._pitch_contours = _scale_contours(pitch(small), sx, sy)
            self._pitch_cache_counter = 0
        pitch_contours = self._pitch_contours

        yolo_ball: Detection | None = None
        yolo_person: Detection | None = None

        if self.detection_mode in ("yolo", "hybrid") and self.yolo and self.yolo.available:
            yolo_ball, yolo_person, _ = self.yolo.detect(small)
            if yolo_ball:
                yolo_ball = Detection(
                    int(yolo_ball.x * sx), int(yolo_ball.y * sy),
                    int(yolo_ball.width * sx), int(yolo_ball.height * sy),
                    yolo_ball.confidence, yolo_ball.class_name,
                )
            if yolo_person:
                yolo_person = Detection(
                    int(yolo_person.x * sx), int(yolo_person.y * sy),
                    int(yolo_person.width * sx), int(yolo_person.height * sy),
                    yolo_person.confidence, yolo_person.class_name,
                )

        ball_x, ball_y = 0, 0
        ball_source = "none"
        ball_conf = 0.0

        if self.detection_mode == "yolo" and yolo_ball:
            ball_x, ball_y = yolo_ball.x, yolo_ball.y
            ball_source = "yolo"
            ball_conf = yolo_ball.confidence
        elif self.detection_mode == "color" and color_x != 0:
            ball_x, ball_y = color_x, color_y
            ball_source = "color"
            ball_conf = 0.72
        elif self.detection_mode == "hybrid":
            if yolo_ball and yolo_ball.confidence >= (self.yolo.ball_conf if self.yolo else 1.0):
                ball_x, ball_y = yolo_ball.x, yolo_ball.y
                ball_source = "yolo"
                ball_conf = yolo_ball.confidence
            elif color_x != 0:
                ball_x, ball_y = color_x, color_y
                ball_source = "color"
                ball_conf = 0.72
            else:
                pred = self.tracker.predict()
                if pred:
                    ball_x, ball_y = pred
                    ball_source = "tracker"
                    ball_conf = 0.45

        if ball_x != 0 or ball_y != 0:
            self.tracker.update(ball_x, ball_y)
        else:
            pred = self.tracker.predict()
            if pred and self.detection_mode == "hybrid":
                ball_x, ball_y = pred
                ball_source = "tracker"
                ball_conf = 0.38

        if yolo_person and yolo_person.confidence >= 0.4:
            h = yolo_person.height
            w = yolo_person.width
            x1 = yolo_person.x - w // 2
            y1 = yolo_person.y - h // 2
            person_contour = np.array(
                [[[x1, y1]], [[x1 + w, y1]], [[x1 + w, y1 + h]], [[x1, y1 + h]]],
                dtype=np.int32,
            )
            batsman_contours = [person_contour] + list(batsman_contours)

        return HybridResult(
            ball_x=ball_x,
            ball_y=ball_y,
            ball_source=ball_source,
            ball_confidence=ball_conf,
            ball_overlay=None,
            batsman_contours=batsman_contours,
            yolo_person=yolo_person,
            pitch_contours=pitch_contours,
        )
