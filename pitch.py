import cv2
import numpy as np


def pitch(img, *, area_min: int = 50000):
    """
    Detect cricket pitch region using HSV green/brown turf segmentation.

    Returns contours sorted by area (largest first). Empty list if none found.
    """
    if img is None or img.size == 0:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Turf / dry pitch — broad range for club grounds
    lower = np.array([25, 25, 40], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    filtered = [c for c in contours if cv2.contourArea(c) >= area_min]
    filtered.sort(key=cv2.contourArea, reverse=True)
    return filtered[:3]
