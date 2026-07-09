# RDK X5 Bottle Hunter

[简体中文](./README_cn.md) | English

An autonomous "bottle hunter" robot built on the Horizon **RDK X5** developer
kit. It uses on-board **YOLOv5** inference (BPU), a **VP100 LiDAR** for obstacle
avoidance, and an **STM32** servo/motor board for locomotion. The robot searches
for bottles with the camera, differential-steers toward the closest one,
performs a spin-forward "capture" when it gets close, and turns away when the
LiDAR sees a wall ahead.

## Features

- **YOLOv5 detection on BPU** – real-time bottle detection using `hobot_dnn`
  and the C++ `libpostprocess.so` post-processing library.
- **LiDAR obstacle avoidance** – forward-sector wall detection from the VP100
  laser scan.
- **STM32 motor control** – 12-byte PWM frame protocol over UART (1000–2000 µs,
  1500 = neutral).
- **Two ways to run**:
  - A standalone monolithic script (`main.py`), and
  - A modular **ROS 2 (Humble)** package set (`src/`).

## State Machine

| State      | Trigger                                   | Behaviour                                  |
| ---------- | ----------------------------------------- | ------------------------------------------ |
| `SEARCH`   | No bottle detected                        | Hold still, wait for a target              |
| `APPROACH` | Bottle visible                            | Differential steering toward the bottle    |
| `CAPTURE`  | Bottle bbox height ≥ capture ratio        | Fixed-duration spin-forward grab           |
| `AVOID`    | LiDAR wall in forward sector              | Fixed-duration turn manoeuvre              |

## Repository Layout

```
RDK_work/
├── main.py                     # Standalone all-in-one controller
├── stm32_servo.py              # STM32 servo UART protocol
├── lidar_cloud_snapshot.py     # Save a top-down LiDAR point cloud PNG
├── scan_relay.py               # /scan (BEST_EFFORT) -> /scan_view (RELIABLE)
├── yolov5_usb_camera_mjpeg.py  # YOLOv5 detection on a USB (MJPEG) camera
├── start_lidar.sh / start_viz.sh
├── best_detect_640x640_bayese_nv12.bin   # Quantized YOLOv5 model
└── src/                        # ROS 2 workspace
    ├── rdk_bottle_hunter/      # Nodes: camera_detector, controller, motor_driver
    ├── rdk_interfaces/         # Custom msgs: BottleDetection, MotorCommand
    └── vp100_ros2/             # VP100 LiDAR driver
```

## Requirements

- Horizon RDK X5 (Ubuntu, ROS 2 Humble)
- `hobot_dnn` runtime and `/usr/lib/libpostprocess.so`
- Python 3: `numpy`, `opencv-python`, `pyserial`, `matplotlib` (for snapshots)
- A V4L2 camera, VP100 LiDAR, and STM32 servo board

## Usage

### 1. Standalone script

```bash
python3 main.py \
    --servo-port /dev/ttyS1 \
    --lidar-port /dev/ttyUSB0 \
    --cam 0
```

Useful options: `--conf` (detection confidence), `--safe-dist` (avoidance
distance in metres), `--capture-ratio` (bbox height ratio that triggers a grab).

### 2. ROS 2 stack

```bash
# Build
colcon build
source install/setup.bash

# Launch the full stack (LiDAR + camera + controller + motor driver)
ros2 launch rdk_bottle_hunter bottle_hunter.launch.py use_lidar:=true
```

Parameters live in `src/rdk_bottle_hunter/config/bottle_hunter.yaml`.

#### ROS 2 topics

| Topic               | Type                              | Description                     |
| ------------------- | --------------------------------- | ------------------------------- |
| `/scan`             | `sensor_msgs/LaserScan`           | VP100 LiDAR scan                |
| `/bottle_detection` | `rdk_interfaces/BottleDetection`  | Best bottle detection per frame |
| `/motor_cmd`        | `rdk_interfaces/MotorCommand`     | PWM command to the STM32 board  |

### LiDAR visualization

```bash
./start_viz.sh                       # lidar driver + QoS relay + rosbridge
python3 lidar_cloud_snapshot.py --frames 5 --out lidar_cloud.png
```

## License

Apache-2.0 (see individual package manifests).
