#!/usr/bin/env python3

import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path


class SensorFusionNode(DTROS):
    """
    Sensor fusion node combining wheel odometry and visual SLAM
    using an Extended Kalman Filter (EKF).

    State vector: [x, y, theta] — 2D robot pose in the map frame.

    Prediction step: odometry delta (differential drive model).
    Update step:     SLAM heading theta (scale-independent correction).

    Subscribers:
        /<veh>/pose       (nav_msgs/Odometry)   — odometry prediction
        /<veh>/slam/pose  (geometry_msgs/PoseStamped) — SLAM correction

    Publishers:
        /<veh>/fusion/pose  (geometry_msgs/PoseStamped) — fused pose
        /<veh>/fusion/path  (nav_msgs/Path)             — fused trajectory
    """

    def __init__(self, node_name: str):
        super().__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.veh = rospy.get_namespace().strip("/")

        # EKF state: [x, y, theta]
        self._x = np.zeros(3)
        self._P = np.diag([0.1, 0.1, 0.1])

        # Process noise (odometry uncertainty)
        self._Q = np.diag([0.01, 0.01, 0.005])

        # Measurement noise (SLAM heading uncertainty — high because
        # camera frame yaw ≠ robot frame yaw without explicit extrinsic calibration)
        self._R_slam = np.array([[1.0]])

        # Innovation gate: ignore SLAM corrections larger than this (rad)
        self._slam_gate = np.deg2rad(30)

        # Previous odometry for delta computation
        self._odom_prev = None

        # Fused path accumulation
        self._path = Path()
        self._path.header.frame_id = "map"

        rospy.Subscriber(
            f"/{self.veh}/pose",
            Odometry,
            self._cb_odometry,
            queue_size=10,
        )
        rospy.Subscriber(
            f"/{self.veh}/slam/pose",
            PoseStamped,
            self._cb_slam,
            queue_size=10,
        )

        self._pub_pose = rospy.Publisher(
            f"/{self.veh}/fusion/pose",
            PoseStamped,
            queue_size=1,
            dt_topic_type=TopicType.LOCALIZATION,
        )
        self._pub_path = rospy.Publisher(
            f"/{self.veh}/fusion/path",
            Path,
            queue_size=1,
        )

        self.log("Initialized.")

    # ------------------------------------------------------------------
    # EKF Prediction — odometry
    # ------------------------------------------------------------------

    def _cb_odometry(self, msg: Odometry):
        if self._odom_prev is None:
            self._odom_prev = msg
            return

        dx     = msg.pose.pose.position.x - self._odom_prev.pose.pose.position.x
        dy     = msg.pose.pose.position.y - self._odom_prev.pose.pose.position.y
        dtheta = self._yaw_from_odom(msg) - self._yaw_from_odom(self._odom_prev)
        dtheta = self._wrap(dtheta)
        self._odom_prev = msg

        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dtheta) < 1e-6:
            return

        d     = np.sqrt(dx ** 2 + dy ** 2)
        theta = self._x[2]

        # State prediction
        self._x[0] += d * np.cos(theta + dtheta / 2)
        self._x[1] += d * np.sin(theta + dtheta / 2)
        self._x[2]  = self._wrap(self._x[2] + dtheta)

        # Covariance prediction
        F = np.array([
            [1, 0, -d * np.sin(theta + dtheta / 2)],
            [0, 1,  d * np.cos(theta + dtheta / 2)],
            [0, 0,  1],
        ])
        self._P = F @ self._P @ F.T + self._Q

        self._publish(msg.header.stamp)

    # ------------------------------------------------------------------
    # EKF Update — SLAM heading correction
    # ------------------------------------------------------------------

    def _cb_slam(self, msg: PoseStamped):
        theta_slam = self._yaw_from_quat(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )

        H = np.array([[0.0, 0.0, 1.0]])
        y = self._wrap(theta_slam - self._x[2])

        # Innovation gate — skip update if SLAM correction is unreliable
        if abs(y) > self._slam_gate:
            return

        S = H @ self._P @ H.T + self._R_slam
        K = (self._P @ H.T @ np.linalg.inv(S)).ravel()

        self._x    += K * y
        self._x[2]  = self._wrap(self._x[2])
        self._P     = (np.eye(3) - np.outer(K, H)) @ self._P

        self._publish(msg.header.stamp)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish(self, stamp):
        x, y, theta = self._x

        pose_msg = PoseStamped()
        pose_msg.header.stamp    = stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.orientation.z = np.sin(theta / 2)
        pose_msg.pose.orientation.w = np.cos(theta / 2)
        self._pub_pose.publish(pose_msg)

        self._path.header.stamp = stamp
        self._path.poses.append(pose_msg)
        self._pub_path.publish(self._path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _yaw_from_odom(msg: Odometry) -> float:
        q = msg.pose.pose.orientation
        return SensorFusionNode._yaw_from_quat(q.x, q.y, q.z, q.w)

    @staticmethod
    def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def on_shutdown(self):
        super().on_shutdown()


if __name__ == "__main__":
    node = SensorFusionNode(node_name="sensor_fusion_node")
    rospy.spin()
