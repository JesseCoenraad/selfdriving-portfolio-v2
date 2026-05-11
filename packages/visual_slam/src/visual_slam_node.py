#!/usr/bin/env python3

import threading

import cv2
import numpy as np
import rospy
import std_msgs.msg
from duckietown.dtros import DTROS, NodeType
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


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
        /<veh>/slam/objects                  (MarkerArray)
    """

    _HSV_RANGES = {
        "duckie": {
            "ranges": [(np.array([15, 80, 80]), np.array([40, 255, 255]))],
            "min_area": 150,
            "color": ColorRGBA(1.0, 1.0, 0.0, 1.0),
        },
        "traffic_light_red": {
            "ranges": [
                (np.array([0, 80, 80]), np.array([15, 255, 255])),
                (np.array([165, 80, 80]), np.array([180, 255, 255])),
            ],
            "min_area": 100,
            "color": ColorRGBA(1.0, 0.1, 0.1, 1.0),
        },
        "traffic_light_green": {
            "ranges": [(np.array([35, 80, 80]), np.array([85, 255, 255]))],
            "min_area": 100,
            "color": ColorRGBA(0.1, 1.0, 0.1, 1.0),
        },
        "duckiebot": {
            "ranges": [(np.array([95, 80, 50]), np.array([135, 255, 255]))],
            "min_area": 300,
            "color": ColorRGBA(0.1, 0.1, 1.0, 1.0),
        },
    }

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

        self._map_objects: list = []

        self._pc_pub  = rospy.Publisher(
            f"/{self.veh}/slam/point_cloud", PointCloud2, queue_size=1
        )
        self._obj_pub = rospy.Publisher(
            f"/{self.veh}/slam/objects", MarkerArray, queue_size=1
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

        rospy.Timer(rospy.Duration(1.0), lambda _: self._publish_cloud())
        rospy.Timer(rospy.Duration(1.0), lambda _: self._publish_objects())

        self.log("Initialized.")

    def _camera_info_cb(self, msg: CameraInfo):
        with self._K_lock:
            if self._K is None:
                self._K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
                self.log("Camera intrinsics loaded from camera_info.")

    def _image_cb(self, msg: CompressedImage):
        buf        = np.frombuffer(msg.data, dtype=np.uint8)
        frame_bgr  = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Object detection on every frame
        objects = self._detect_objects(frame_bgr)
        if objects:
            self._add_objects(objects)
            self._publish_objects()

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

        pts1 = np.float32([self._prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        if pts1.shape[0] < 8 or pts2.shape[0] < 8:
            self._prev_kp, self._prev_desc = kp, desc
            return

        # Fundamental matrix with RANSAC to filter outliers
        try:
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
        except cv2.error:
            self._prev_kp, self._prev_desc = kp, desc
            return

        if F is None or mask is None or F.shape != (3, 3):
            self._prev_kp, self._prev_desc = kp, desc
            return

        pts1_in = pts1[mask.ravel() == 1].reshape(-1, 2)
        pts2_in = pts2[mask.ravel() == 1].reshape(-1, 2)
        if len(pts1_in) < 8:
            self._prev_kp, self._prev_desc = kp, desc
            return

        # Essential matrix → relative pose
        try:
            E = K.T @ F @ K
            _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
        except cv2.error:
            self._prev_kp, self._prev_desc = kp, desc
            return

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

    def _detect_objects(self, frame_bgr: np.ndarray) -> list:
        hsv     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        results = []
        for label, cfg in self._HSV_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in cfg["ranges"]:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < cfg["min_area"]:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                results.append({"type": label, "pixel": (cx, cy)})
        return results

    def _add_objects(self, detections: list):
        with self._K_lock:
            K = self._K if self._K is not None else self._default_K
        cam_pos = self._t_cw.ravel()
        for obj in detections:
            px, py = obj["pixel"]
            p_c = np.array([
                (px - K[0, 2]) / K[0, 0],
                (py - K[1, 2]) / K[1, 1],
                1.0,
            ])
            p_w = self._R_cw @ p_c + cam_pos
            self._map_objects.append({"type": obj["type"], "position": p_w.tolist()})
        if len(self._map_objects) > 500:
            self._map_objects = self._map_objects[-500:]

    def _publish_objects(self):
        markers = MarkerArray()
        now     = rospy.Time.now()
        for i, obj in enumerate(self._map_objects[-100:]):
            p     = obj["position"]
            label = obj["type"]
            color = self._HSV_RANGES[label]["color"]
            m = Marker()
            m.header.stamp       = now
            m.header.frame_id    = "map"
            m.ns                 = "objects"
            m.id                 = i
            m.type               = Marker.SPHERE
            m.action             = Marker.ADD
            m.pose.position.x    = p[0]
            m.pose.position.y    = p[1]
            m.pose.position.z    = p[2]
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color  = color
            markers.markers.append(m)
        self._obj_pub.publish(markers)

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
