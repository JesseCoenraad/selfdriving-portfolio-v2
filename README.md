# selfdriving-portfolio-v2

A ROS-based self-driving stack for Duckietown, implementing:

- **Wheel-encoder odometry** (`encoder_pose`) — dead-reckoning pose estimation from left/right encoder ticks.
- **Monocular visual SLAM** (`visual_slam`) — Shi-Tomasi corner tracking via Lucas-Kanade optical flow, essential-matrix motion estimation, and a live landmark map with Duckietown object detection.

Built on top of [`dt-core`](https://github.com/duckietown/dt-core).

---

## Running on a real Duckiebot

```bash
dts devel run -H <hostname>.local -L slam
```

---

## Running in the Duckiematrix (simulation)

The Duckiematrix provides a simulated Duckietown environment with virtual cameras
that publish on the same ROS topics as a real robot.

### 1. Build the image for amd64

```bash
dts devel build -a amd64
```

### 2. Start the Duckiematrix

```bash
dts matrix start
```

### 3. Run the container against the simulator

```bash
dts devel run -a amd64 -L slam --sim
```

---

## Packages

| Package | Launcher | Description |
|---|---|---|
| `encoder_pose` | `odometry` | Dead-reckoning pose from wheel encoders |
| `visual_slam` | `slam` | Monocular SLAM with feature tracking and object detection |

The default launcher runs both nodes together.

---

## ROS topics

### encoder_pose

| Direction | Topic | Type |
|---|---|---|
| Sub | `/<veh>/left_wheel_encoder_driver_node/tick` | `WheelEncoderStamped` |
| Sub | `/<veh>/right_wheel_encoder_driver_node/tick` | `WheelEncoderStamped` |
| Pub | `/<veh>/pose` | `Odometry` |

### visual_slam

| Direction | Topic | Type |
|---|---|---|
| Sub | `/<veh>/camera_node/image/compressed` | `CompressedImage` |
| Pub | `/<veh>/slam/pose` | `PoseStamped` |
| Pub | `/<veh>/slam/map` | `MarkerArray` |
| Pub | `/<veh>/slam/debug/image/compressed` | `CompressedImage` |