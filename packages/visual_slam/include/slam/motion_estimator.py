from typing import Optional, Tuple

import cv2
import numpy as np


class MotionEstimator:
    """
    Estimates relative camera motion between two frames using the essential matrix.

    Returns R, t up to scale (monocular ambiguity). Scale is resolved externally
    by the sensor fusion component (Task 3).
    """

    def __init__(self, K: np.ndarray):
        self.K = K

    def estimate(
        self, pts1: np.ndarray, pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Args:
            pts1: matched points in previous frame, shape (N, 1, 2) or (N, 2)
            pts2: matched points in current frame, same shape

        Returns:
            (R, t): rotation matrix (3x3) and translation vector (3x1),
                    or (None, None) if estimation fails.
        """
        p1 = pts1.reshape(-1, 2).astype(np.float64)
        p2 = pts2.reshape(-1, 2).astype(np.float64)

        if len(p1) < 8:
            return None, None

        E, mask = cv2.findEssentialMat(
            p1, p2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            return None, None

        _, R, t, _ = cv2.recoverPose(E, p1, p2, self.K, mask=mask)
        return R, t
