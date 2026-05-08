from typing import Tuple

import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> float:
    """
    Args:
        ticks: Current tick count from the encoders.
        prev_ticks: Previous tick count from the encoders.
        resolution: Number of ticks per full wheel rotation returned by the encoder.
    Return:
        dphi: Rotation of the wheel in radians.
    """
    dphi = (ticks - prev_ticks) / resolution * 2 * np.pi
    return dphi


def estimate_pose(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:
    """
    Calculate the current Duckiebot pose using the dead-reckoning model.

    Args:
        R:                  radius of wheel (both wheels assumed equal)
        baseline:           distance between wheels (2L)
        x_prev:             previous x estimate
        y_prev:             previous y estimate
        theta_prev:         previous orientation estimate
        delta_phi_left:     left wheel rotation (rad)
        delta_phi_right:    right wheel rotation (rad)

    Return:
        x_curr:      estimated x coordinate
        y_curr:      estimated y coordinate
        theta_curr:  estimated heading
    """
    d_left = delta_phi_left * R
    d_right = delta_phi_right * R

    d = (d_left + d_right) / 2
    dtheta = (d_right - d_left) / baseline

    theta_curr = theta_prev + dtheta
    x_curr = x_prev + d * np.cos(theta_prev + dtheta / 2)
    y_curr = y_prev + d * np.sin(theta_prev + dtheta / 2)

    return x_curr, y_curr, theta_curr
