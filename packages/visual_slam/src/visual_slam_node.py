#!/usr/bin/env python3
from typing import List, Dict, Optional

import cv2
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from visual_slam.include.slam.feature_tracker import FeatureTracker
from visual_slam.include.slam.motion_estimator import MotionEstimator
from visual_slam.include.slam.object_detector import ObjectDetector


class VisualSlamNode(DTROS):
    """
    Vision-based monocular SLAM node.

    Tracks Shi-Tomasi corners via Lucas-Kanade optical flow, estimates
    relative camera motion via the essential matrix, and builds a live
    map of feature landmarks and detected Duckietown objects.

    Scale is unit (monocular ambiguity); sensor fusion with odometry
    resolves metric scale in Task 3.

    Subscribers:
        /<veh>/camera_node/image/compressed  (CompressedImage)

    Publishers:
        /<veh>/slam/pose                     (PoseStamped)
        /<veh>/slam/map                      (MarkerArray)
        /<veh>/slam/debug/image/compressed   (CompressedImage)
    """

    # Duckietown DB21 camera defaults — overridden by calibration if present
    _DEFAULT_K = np.array(
        [[313.8, 0.0, 321.5], [0.0, 313.8, 237.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    _OBJ_COLORS: Dict[str, ColorRGBA] = {
        "duckie":              ColorRGBA(1.0, 1.0, 0.0, 1.0),
        "traffic_light_red":   ColorRGBA(1.0, 0.1, 0.1, 1.0),
        "traffic_light_green": ColorRGBA(0.1, 1.0, 0.1, 1.0),
    }

    _DBG_COLORS: Dict[str, tuple] = {
        "duckie":              (0, 255, 255),
        "traffic_light_red":   (0, 0, 255),
        "traffic_light_green": (0, 255, 0),
    }

    MAX_MAP_FEATURES = 2000
    MAX_MAP_OBJECTS  = 500

    def __init__(self, node_name: str):
        super().__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.veh = rospy.get_namespace().strip("/")

        self.K = self._load_camera_matrix()
        self.tracker  = FeatureTracker()
        self.estimator = MotionEstimator(self.K)
        self.detector  = ObjectDetector()

        # SLAM state
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts:  Optional[np.ndarray] = None
        self.pose = np.eye(4, dtype=np.float64)   # accumulated camera pose in world frame
        self.map_features: List[List[float]] = []  # [x, y, z] world points
        self.map_objects:  List[Dict]        = []  # {type, position}

        rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.cb_image,
            queue_size=1,
            buff_size=2 ** 24,
        )

        self.pub_pose  = rospy.Publisher(f"/{self.veh}/slam/pose", PoseStamped,  queue_size=1)
        self.pub_map   = rospy.Publisher(f"/{self.veh}/slam/map",  MarkerArray,  queue_size=1)
        self.pub_debug = rospy.Publisher(
            f"/{self.veh}/slam/debug/image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG,
        )

        self.log("Initialized.")

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def cb_image(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = self.detector.detect(frame)

        if self.prev_gray is None:
            self.prev_pts  = self.tracker.detect(gray)
            self.prev_gray = gray
            self._publish_debug(frame, self.prev_pts, objects)
            return

        prev_good, curr_good = self.tracker.track(self.prev_gray, gray, self.prev_pts)

        R, t = self.estimator.estimate(prev_good, curr_good)
        if R is not None:
            self._update_pose(R, t)
            self._add_features(curr_good)
            self._add_objects(objects)
            self._publish_pose(msg.header.stamp)
            self._publish_map()

        # Re-detect features when count drops below threshold
        if self.tracker.needs_redetect(curr_good):
            self.prev_pts = self.tracker.detect(gray)
        else:
            self.prev_pts = curr_good.reshape(-1, 1, 2)

        self.prev_gray = gray
        self._publish_debug(frame, self.prev_pts, objects)

    # ------------------------------------------------------------------
    # SLAM state updates
    # ------------------------------------------------------------------

    def _update_pose(self, R: np.ndarray, t: np.ndarray):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = t.ravel()
        self.pose = self.pose @ np.linalg.inv(T)

    def _add_features(self, pts: np.ndarray):
        cam_pos = self.pose[:3, 3]
        for pt in pts.reshape(-1, 2):
            # Back-project to unit depth in camera frame, then transform to world
            p_c = np.array([
                (pt[0] - self.K[0, 2]) / self.K[0, 0],
                (pt[1] - self.K[1, 2]) / self.K[1, 1],
                1.0,
            ])
            p_w = self.pose[:3, :3] @ p_c + cam_pos
            self.map_features.append(p_w.tolist())
        if len(self.map_features) > self.MAX_MAP_FEATURES:
            self.map_features = self.map_features[-self.MAX_MAP_FEATURES:]

    def _add_objects(self, detections: List[Dict]):
        cam_pos = self.pose[:3, 3]
        for obj in detections:
            px, py = obj["pixel"]
            p_c = np.array([
                (px - self.K[0, 2]) / self.K[0, 0],
                (py - self.K[1, 2]) / self.K[1, 1],
                1.0,
            ])
            p_w = self.pose[:3, :3] @ p_c + cam_pos
            self.map_objects.append({"type": obj["type"], "position": p_w.tolist()})
        if len(self.map_objects) > self.MAX_MAP_OBJECTS:
            self.map_objects = self.map_objects[-self.MAX_MAP_OBJECTS:]

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _publish_pose(self, stamp):
        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = "map"
        p = self.pose[:3, 3]
        msg.pose.position.x = p[0]
        msg.pose.position.y = p[1]
        msg.pose.position.z = p[2]
        q = self._rot_to_quat(self.pose[:3, :3])
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.pub_pose.publish(msg)

    def _publish_map(self):
        markers = MarkerArray()
        now = rospy.Time.now()

        # Feature points as a POINTS marker (efficient single message)
        if self.map_features:
            m = Marker()
            m.header.stamp    = now
            m.header.frame_id = "map"
            m.ns     = "features"
            m.id     = 0
            m.type   = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.02
            m.scale.y = 0.02
            m.color   = ColorRGBA(0.2, 0.5, 1.0, 0.5)
            for p in self.map_features[-500:]:
                m.points.append(Point(x=p[0], y=p[1], z=p[2]))
            markers.markers.append(m)

        # Detected objects as individual sphere markers
        for i, obj in enumerate(self.map_objects[-100:]):
            p = obj["position"]
            color = self._OBJ_COLORS.get(obj["type"], ColorRGBA(1.0, 1.0, 1.0, 1.0))
            m = Marker()
            m.header.stamp       = now
            m.header.frame_id    = "map"
            m.ns                 = "objects"
            m.id                 = i + 1
            m.type               = Marker.SPHERE
            m.action             = Marker.ADD
            m.pose.position.x    = p[0]
            m.pose.position.y    = p[1]
            m.pose.position.z    = p[2]
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.12
            m.color = color
            markers.markers.append(m)

        self.pub_map.publish(markers)

    def _publish_debug(self, frame: np.ndarray, pts: Optional[np.ndarray], objects: List[Dict]):
        if self.pub_debug.get_num_connections() == 0:
            return

        debug = frame.copy()

        if pts is not None:
            for pt in pts.reshape(-1, 2):
                cv2.circle(debug, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        for obj in objects:
            px, py  = obj["pixel"]
            label   = obj["type"]
            color   = self._DBG_COLORS.get(label, (255, 255, 255))
            cv2.circle(debug, (px, py), 15, color, 2)
            cv2.putText(debug, label, (px + 5, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        _, enc = cv2.imencode(".jpg", debug, [cv2.IMWRITE_JPEG_QUALITY, 70])
        out = CompressedImage()
        out.header.stamp = rospy.Time.now()
        out.format = "jpeg"
        out.data   = enc.tobytes()
        self.pub_debug.publish(out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_camera_matrix(self) -> np.ndarray:
        import os, yaml
        calib_path = f"/data/config/calibrations/camera_intrinsic/{self.veh}.yaml"
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                data = yaml.safe_load(f)
            K_flat = data["camera_matrix"]["data"]
            self.log(f"Loaded camera calibration from {calib_path}")
            return np.array(K_flat, dtype=np.float64).reshape(3, 3)
        self.log("No calibration file found — using default camera matrix.")
        return self._DEFAULT_K.copy()

    @staticmethod
    def _rot_to_quat(R: np.ndarray) -> np.ndarray:
        """Rotation matrix → quaternion [x, y, z, w] via Shepperd's method."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])

    def on_shutdown(self):
        super().on_shutdown()


if __name__ == "__main__":
    node = VisualSlamNode(node_name="visual_slam_node")
    rospy.spin()
