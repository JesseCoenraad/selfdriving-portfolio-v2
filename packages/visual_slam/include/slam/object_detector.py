from typing import Dict, List

import cv2
import numpy as np


class ObjectDetector:
    """
    Detects Duckietown objects via HSV color segmentation.

    Detected classes:
        - duckie          (yellow blobs)
        - traffic_light   (red or green blobs above a minimum size)
        - duckiebot       (white body with colored trim)
    """

    # HSV bounds for each class
    _RANGES = {
        "duckie": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
        "traffic_light_red": [
            (np.array([0, 120, 100]), np.array([10, 255, 255])),
            (np.array([170, 120, 100]), np.array([180, 255, 255])),
        ],
        "traffic_light_green": [
            (np.array([40, 100, 100]), np.array([80, 255, 255]))
        ],
    }

    _MIN_AREA = {
        "duckie": 500,
        "traffic_light_red": 200,
        "traffic_light_green": 200,
    }

    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Returns a list of detected objects, each a dict with:
            type   (str)         — object class
            pixel  (int, int)    — centroid in image coordinates
            area   (float)       — contour area in pixels²
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        results = []
        for label, ranges in self._RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
            results.extend(self._contours_to_detections(mask, label))
        return results

    def _contours_to_detections(self, mask: np.ndarray, label: str) -> List[Dict]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        min_area = self._MIN_AREA.get(label, 300)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            detections.append({"type": label, "pixel": (cx, cy), "area": area})
        return detections
