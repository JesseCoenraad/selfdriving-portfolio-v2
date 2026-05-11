#!/usr/bin/env python3

import threading

import cv2
import numpy as np
import rospy
import std_msgs.msg
from duckietown.dtros import DTROS, NodeType
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField


class VisualSlamNode(DTROS):
    """
    ORB-SLAM pipeline node.

    Subscribes to a compressed camera image stream, runs ORB feature extraction
    and matching on every consecutive pair of frames, estimates relative pose via
    the essential matrix, triangulates 3-D landmarks, and publishes a growing
    point cloud on /<veh>/slam/point_cloud.

    Subscribers:
        /<veh>/camera_node/image/compressed  (CompressedImage)
        /<veh>/camera_node/camera_info       (CameraInfo)

    Publishers:
        /<veh>/slam/point_cloud              (PointCloud2)
    """

    def __init__(self, node_name: str):
        super().__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.veh = rospy.get_namespace().strip("/")

        n_features = rospy.get_param("~n_features", 5000)
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Camera intrinsics — updated from camera_info when available
        self._K_lock = threading.Lock()
        self._K = None
        k_flat = rospy.get_param(
            "~camera_matrix",
            [313.8, 0.0, 321.5, 0.0, 313.8, 237.5, 0.0, 0.0, 1.0],
        )
        self._default_K = np.array(k_flat, dtype=np.float64).reshape(3, 3)

        # Previous-frame ORB state
        self._prev_kp   = None
        self._prev_desc = None

        # Accumulated global pose (camera-to-world)
        self._R_cw = np.eye(3)
        self._t_cw = np.zeros((3, 1))

        # Map accumulation
        self._map_lock = threading.Lock()
        self._map_pts: list = []
        self._max_pts: int  = rospy.get_param("~max_map_points", 10000)

        self._pc_pub = rospy.Publisher(
            f"/{self.veh}/slam/point_cloud", PointCloud2, queue_size=1
        )

        rospy.Subscriber(
            f"/{self.veh}/camera_node/camera_info",
            CameraInfo,
            self._camera_info_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self._image_cb,
            queue_size=1,
            buff_size=2 ** 24,
        )

        self.log("Initialized.")

    def _camera_info_cb(self, msg: CameraInfo):
        with self._K_lock:
            if self._K is None:
                self._K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
                self.log("Camera intrinsics loaded from camera_info.")

    def _image_cb(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return

        kp, desc = self._orb.detectAndCompute(frame, None)
        if desc is None or len(kp) < 8:
            self._prev_kp, self._prev_desc = kp, desc
            return

        if self._prev_kp is None or self._prev_desc is None:
            self._prev_kp, self._prev_desc = kp, desc
            return

        with self._K_lock:
            K = self._K if self._K is not None else self._default_K

        # Feature matching
        matches = sorted(
            self._bf.match(self._prev_desc, desc), key=lambda m: m.distance
        )
        if len(matches) < 8:
            self._prev_kp, self._prev_desc = kp, desc
            return

        pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        # Fundamental matrix with RANSAC to filter outliers
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        if F is None or mask is None:
            self._prev_kp, self._prev_desc = kp, desc
            return

        pts1_in = pts1[mask.ravel() == 1]
        pts2_in = pts2[mask.ravel() == 1]
        if len(pts1_in) < 5:
            self._prev_kp, self._prev_desc = kp, desc
            return

        # Essential matrix → relative pose
        E = K.T @ F @ K
        _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)

        # Triangulate in the previous camera frame
        p1_n = cv2.undistortPoints(pts1_in.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        p2_n = cv2.undistortPoints(pts2_in.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))
        pts_4d = cv2.triangulatePoints(P1, P2, p1_n.T, p2_n.T).T

        # Dehomogenize and discard near-zero-weight points
        w       = pts_4d[:, 3:4]
        valid_w = np.abs(w.ravel()) > 1e-8
        pts_4d  = pts_4d[valid_w] / w[valid_w]
        pts_local = pts_4d[:, :3]

        # Keep only points in front of both cameras (positive depth)
        pts_cam2 = (R @ pts_local.T + t).T
        good      = (pts_local[:, 2] > 0) & (pts_cam2[:, 2] > 0)
        pts_local = pts_local[good]

        if len(pts_local) > 0:
            pts_world = (self._R_cw @ pts_local.T).T + self._t_cw.T
            with self._map_lock:
                self._map_pts.extend(pts_world.tolist())
                if len(self._map_pts) > self._max_pts:
                    self._map_pts = self._map_pts[-self._max_pts:]
            self._publish_cloud()

        # Update global pose
        R_cw_new   = self._R_cw @ R.T
        self._t_cw = self._t_cw - R_cw_new @ t
        self._R_cw = R_cw_new

        self._prev_kp, self._prev_desc = kp, desc

    def _publish_cloud(self):
        with self._map_lock:
            pts = np.array(self._map_pts, dtype=np.float32)

        header          = std_msgs.msg.Header()
        header.stamp    = rospy.Time.now()
        header.frame_id = "map"

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        self._pc_pub.publish(pc2.create_cloud(header, fields, pts))

    def on_shutdown(self):
        super().on_shutdown()


if __name__ == "__main__":
    node = VisualSlamNode(node_name="visual_slam_node")
    rospy.spin()
