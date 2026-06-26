"""Convert OpenCV BGR numpy array to QImage."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    if frame is None or frame.size == 0:
        return QImage()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
