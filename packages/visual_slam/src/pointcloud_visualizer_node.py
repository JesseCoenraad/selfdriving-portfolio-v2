#!/usr/bin/env python3
"""
PointCloud visualizer ROS node.

Subscribes to /<veh>/slam/point_cloud (sensor_msgs/PointCloud2) and renders the
points in a live window. Open3D is used when available (best quality); the node
falls back to a matplotlib 3-D scatter plot otherwise.

The Open3D event loop and the matplotlib draw loop both require the main thread,
so rospy.spin() is dispatched to a daemon thread.

ROS parameters
~~~~~~~~~~~~~~
~point_cloud_topic : input topic          (default: /vehicle/slam/point_cloud)
~max_display_pts   : subsample cap        (default: 8000)
~point_size        : rendered point size  (default: 2.0)
"""

import threading

import numpy as np
import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

try:
    import open3d as o3d
    _BACKEND = "open3d"
except ImportError:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _BACKEND = "matplotlib"


class _Open3DVisualizer:
    def __init__(self, max_pts: int, point_size: float):
        self._max_pts  = max_pts
        self._lock     = threading.Lock()
        self._pts      = np.empty((0, 3), dtype=np.float64)
        self._updated  = False

        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window("ORB-SLAM Point Cloud", width=1280, height=720)
        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)

        opt = self._vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = point_size

    def update(self, pts: np.ndarray):
        with self._lock:
            self._pts     = pts
            self._updated = True

    def spin_once(self) -> bool:
        with self._lock:
            if self._updated:
                pts = self._pts
                self._updated = False
                if len(pts) > self._max_pts:
                    idx = np.random.choice(len(pts), self._max_pts, replace=False)
                    pts = pts[idx]
                self._pcd.points = o3d.utility.Vector3dVector(pts)
                colors = np.zeros((len(pts), 3))
                colors[:, 1] = 1.0  # green
                self._pcd.colors = o3d.utility.Vector3dVector(colors)
                self._vis.update_geometry(self._pcd)
        return self._vis.poll_events() and self._vis.update_renderer()

    def destroy(self):
        self._vis.destroy_window()


class _MatplotlibVisualizer:
    def __init__(self, max_pts: int, point_size: float):
        self._max_pts   = max_pts
        self._point_size = max(1, int(point_size))
        self._lock      = threading.Lock()
        self._pts       = np.empty((0, 3), dtype=np.float64)
        self._updated   = False

        plt.ion()
        self._fig = plt.figure("ORB-SLAM Point Cloud", figsize=(12, 8))
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._fig.patch.set_facecolor("#0d0d0d")
        self._ax.set_facecolor("#0d0d0d")
        self._style_axes()

    def _style_axes(self):
        self._ax.set_title("ORB-SLAM Point Cloud", color="white")
        self._ax.set_xlabel("X (m)", color="white")
        self._ax.set_ylabel("Y (m)", color="white")
        self._ax.set_zlabel("Z (m)", color="white")
        self._ax.tick_params(colors="white")

    def update(self, pts: np.ndarray):
        with self._lock:
            self._pts     = pts
            self._updated = True

    def spin_once(self) -> bool:
        if not plt.fignum_exists(self._fig.number):
            return False
        with self._lock:
            if not self._updated:
                self._fig.canvas.flush_events()
                return True
            pts           = self._pts
            self._updated = False
        self._ax.cla()
        self._style_axes()
        if len(pts) > 0:
            if len(pts) > self._max_pts:
                idx = np.random.choice(len(pts), self._max_pts, replace=False)
                pts = pts[idx]
            self._ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c="lime", s=self._point_size, alpha=0.5, marker=".",
            )
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        return True

    def destroy(self):
        plt.close(self._fig)


class PointCloudVisualizerNode:
    def __init__(self):
        rospy.init_node("pointcloud_visualizer_node", anonymous=False)

        max_pts    = rospy.get_param("~max_display_pts", 8000)
        point_size = rospy.get_param("~point_size", 2.0)

        if _BACKEND == "open3d":
            self._viz = _Open3DVisualizer(max_pts, point_size)
            rospy.loginfo("[PointCloudVisualizerNode] Using Open3D backend.")
        else:
            self._viz = _MatplotlibVisualizer(max_pts, point_size)
            rospy.loginfo("[PointCloudVisualizerNode] Using matplotlib backend (open3d not found).")

        rospy.Subscriber(
            rospy.get_param("~point_cloud_topic", "/vehicle/slam/point_cloud"),
            PointCloud2,
            self._callback,
            queue_size=1,
        )
        rospy.loginfo("[PointCloudVisualizerNode] Ready.")

    def _callback(self, msg: PointCloud2):
        raw = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if raw:
            self._viz.update(np.array(raw, dtype=np.float64))

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if not self._viz.spin_once():
                rospy.loginfo("[PointCloudVisualizerNode] Window closed — shutting down.")
                rospy.signal_shutdown("Window closed.")
                break
            rate.sleep()
        self._viz.destroy()


if __name__ == "__main__":
    node = PointCloudVisualizerNode()
    spin_thread = threading.Thread(target=rospy.spin, daemon=True)
    spin_thread.start()
    node.run()
