"""YOLO-based object detection for ball and batsman."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    class_name: str
    source: str = "yolo"


class YOLODetector:
    """Ultralytics YOLOv8 detector for sports ball and person classes."""

    BALL_CLASS = "sports ball"
    PERSON_CLASS = "person"

    def __init__(self, model_path: str = "yolov8n.pt", ball_conf: float = 0.25, person_conf: float = 0.4):
        self.model_path = model_path
        self.ball_conf = ball_conf
        self.person_conf = person_conf
        self._model = None
        self._available = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            self._available = True
        except Exception:
            self._model = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, img: np.ndarray) -> tuple[Detection | None, Detection | None, list]:
        if not self._available or self._model is None or img is None:
            return None, None, []

        results = self._model(img, verbose=False)
        ball: Detection | None = None
        person: Detection | None = None
        all_dets: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                det = Detection(cx, cy, w, h, conf, name)
                all_dets.append(det)

                if name == self.BALL_CLASS and conf >= self.ball_conf:
                    if ball is None or conf > ball.confidence:
                        ball = det
                elif name == self.PERSON_CLASS and conf >= self.person_conf:
                    if person is None or (y2 - y1) > person.height:
                        person = det

        return ball, person, all_dets
