import cv2
import numpy as np


class FeatureTracker:
    """Detects Shi-Tomasi corners and tracks them via Lucas-Kanade optical flow."""

    MIN_FEATURES = 50

    def __init__(self):
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self._detect_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )

    def detect(self, gray: np.ndarray) -> np.ndarray:
        """Returns (N, 1, 2) float32 array of detected corner points."""
        pts = cv2.goodFeaturesToTrack(gray, **self._detect_params)
        return pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)

    def track(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray, prev_pts: np.ndarray
    ):
        """
        Tracks prev_pts from prev_gray into curr_gray.

        Returns:
            (prev_good, curr_good): matched point pairs that passed the LK status check.
        """
        if len(prev_pts) == 0:
            empty = np.empty((0, 1, 2), dtype=np.float32)
            return empty, empty

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self._lk_params
        )
        mask = status.ravel().astype(bool)
        return prev_pts[mask], curr_pts[mask]

    def needs_redetect(self, pts: np.ndarray) -> bool:
        return len(pts) < self.MIN_FEATURES
