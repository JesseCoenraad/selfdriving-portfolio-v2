#!/usr/bin/env python3
import os
from typing import Optional
from multiprocessing import Lock

import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import WheelEncoderStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

from encoder_pose.include.odometry.odometry import delta_phi, estimate_pose


class EncoderPoseNode(DTROS):
    """
    Estimates the Duckiebot pose using wheel encoder data (dead-reckoning).

    Subscribers:
        /<veh>/left_wheel_encoder_driver_node/tick  (WheelEncoderStamped)
        /<veh>/right_wheel_encoder_driver_node/tick (WheelEncoderStamped)

    Publishers:
        /<veh>/pose (Odometry)
    """

    right_tick_prev: Optional[int]
    left_tick_prev: Optional[int]
    delta_phi_left: float
    delta_phi_right: float

    def __init__(self, node_name):
        super(EncoderPoseNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.log("Initializing...")

        self.veh = rospy.get_namespace().strip("/")

        self.left_wheel_mutex = Lock()
        self.right_wheel_mutex = Lock()

        self._reset_state()

        # Wheel geometry (DB21 defaults)
        self.R = 0.0318       # wheel radius in meters
        self.baseline = 0.11  # wheel-to-wheel distance in meters

        left_encoder_topic = f"/{self.veh}/left_wheel_encoder_driver_node/tick"
        right_encoder_topic = f"/{self.veh}/right_wheel_encoder_driver_node/tick"

        rospy.Subscriber(left_encoder_topic, WheelEncoderStamped, self.cb_left_encoder)
        rospy.Subscriber(right_encoder_topic, WheelEncoderStamped, self.cb_right_encoder)

        self.pub_pose = rospy.Publisher(
            f"/{self.veh}/pose",
            Odometry,
            queue_size=1,
            dt_topic_type=TopicType.LOCALIZATION,
        )
        self.pub_path = rospy.Publisher(
            f"/{self.veh}/odometry/path",
            Path,
            queue_size=1,
        )
        self._path = Path()
        self._path.header.frame_id = "map"

        rospy.Timer(rospy.Duration(0.5), self.publish_pose)
        self.log("Initialized.")

    def _reset_state(self):
        self.left_tick_prev = None
        self.right_tick_prev = None
        self.delta_phi_left = 0.0
        self.delta_phi_right = 0.0
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.theta_prev = 0.0

    def cb_left_encoder(self, msg: WheelEncoderStamped):
        with self.left_wheel_mutex:
            if self.left_tick_prev is None:
                self.left_tick_prev = msg.data
                return
            dphi = delta_phi(msg.data, self.left_tick_prev, msg.resolution)
            if dphi == 0:
                return
            self.left_tick_prev = msg.data
            self.delta_phi_left += dphi

    def cb_right_encoder(self, msg: WheelEncoderStamped):
        with self.right_wheel_mutex:
            if self.right_tick_prev is None:
                self.right_tick_prev = msg.data
                return
            dphi = delta_phi(msg.data, self.right_tick_prev, msg.resolution)
            if dphi == 0:
                return
            self.right_tick_prev = msg.data
            self.delta_phi_right += dphi

    def publish_pose(self, event=None):
        with self.left_wheel_mutex:
            with self.right_wheel_mutex:
                x, y, theta = estimate_pose(
                    self.R,
                    self.baseline,
                    self.x_prev,
                    self.y_prev,
                    self.theta_prev,
                    self.delta_phi_left,
                    self.delta_phi_right,
                )

                if x == self.x_prev and y == self.y_prev and theta == self.theta_prev:
                    return

                theta = self._clamp_angle(theta)

                self.log(f"Pose — x: {x:.4f} m  y: {y:.4f} m  θ: {np.rad2deg(theta):.2f}°")

                self.delta_phi_left = 0.0
                self.delta_phi_right = 0.0

                self.x_prev = x
                self.y_prev = y
                self.theta_prev = theta

                odom = Odometry()
                odom.header.frame_id = "map"
                odom.header.stamp = rospy.Time.now()
                odom.pose.pose.position.x = x
                odom.pose.pose.position.y = y
                odom.pose.pose.position.z = 0.0
                odom.pose.pose.orientation.x = 0.0
                odom.pose.pose.orientation.y = 0.0
                odom.pose.pose.orientation.z = np.sin(theta / 2)
                odom.pose.pose.orientation.w = np.cos(theta / 2)

                self.pub_pose.publish(odom)

                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "map"
                pose_stamped.header.stamp = odom.header.stamp
                pose_stamped.pose = odom.pose.pose
                self._path.header.stamp = odom.header.stamp
                self._path.poses.append(pose_stamped)
                self.pub_path.publish(self._path)

    @staticmethod
    def _clamp_angle(theta: float) -> float:
        if theta > 2 * np.pi:
            return theta - 2 * np.pi
        elif theta < -2 * np.pi:
            return theta + 2 * np.pi
        return theta

    def on_shutdown(self):
        super(EncoderPoseNode, self).on_shutdown()


if __name__ == "__main__":
    node = EncoderPoseNode(node_name="encoder_pose_node")
    rospy.spin()
