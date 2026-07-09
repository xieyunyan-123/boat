# RDK X5 捡瓶机器人（Bottle Hunter）

简体中文 | [English](./README.md)

基于地平线 **RDK X5** 开发板的自主"捡瓶"机器人。它使用板载 **YOLOv5** 推理
（BPU）识别瓶子，使用 **VP100 激光雷达** 进行避障，并通过 **STM32** 舵机/电机板
驱动运动。机器人用摄像头搜索瓶子，向最近的目标做差速转向靠近，接近时执行
"旋转前进"捕获动作，当激光雷达检测到前方墙壁时自动转向避障。

## 功能特性

- **BPU 上的 YOLOv5 检测**：基于 `hobot_dnn` 与 C++ 后处理库
  `libpostprocess.so` 实现实时瓶子检测。
- **激光雷达避障**：从 VP100 扫描数据中检测前方扇区内的墙壁/障碍物。
- **STM32 电机控制**：通过 UART 使用 12 字节 PWM 帧协议（1000–2000 µs，
  1500 为中位）。
- **两种运行方式**：
  - 独立的单文件脚本（`main.py`）；
  - 模块化的 **ROS 2（Humble）** 功能包（`src/`）。

## 状态机

| 状态       | 触发条件                        | 行为                        |
| ---------- | ------------------------------- | --------------------------- |
| `SEARCH`   | 未检测到瓶子                    | 原地等待目标                |
| `APPROACH` | 检测到瓶子                      | 差速转向靠近瓶子            |
| `CAPTURE`  | 瓶子框高 ≥ 捕获比例             | 固定时长旋转前进捕获        |
| `AVOID`    | 激光雷达在前方扇区发现墙壁      | 固定时长转向避障            |

## 目录结构

```
RDK_work/
├── main.py                     # 独立的一体化控制器
├── stm32_servo.py              # STM32 舵机 UART 协议
├── lidar_cloud_snapshot.py     # 保存俯视激光点云 PNG
├── scan_relay.py               # /scan (BEST_EFFORT) -> /scan_view (RELIABLE)
├── yolov5_usb_camera_mjpeg.py  # 在 USB（MJPEG）摄像头上运行 YOLOv5
├── start_lidar.sh / start_viz.sh
├── best_detect_640x640_bayese_nv12.bin   # 量化后的 YOLOv5 模型
└── src/                        # ROS 2 工作空间
    ├── rdk_bottle_hunter/      # 节点：camera_detector、controller、motor_driver
    ├── rdk_interfaces/         # 自定义消息：BottleDetection、MotorCommand
    └── vp100_ros2/             # VP100 激光雷达驱动
```

## 环境依赖

- 地平线 RDK X5（Ubuntu、ROS 2 Humble）
- `hobot_dnn` 运行库以及 `/usr/lib/libpostprocess.so`
- Python 3：`numpy`、`opencv-python`、`pyserial`、`matplotlib`（用于点云快照）
- V4L2 摄像头、VP100 激光雷达、STM32 舵机板

## 使用方法

### 1. 独立脚本

```bash
python3 main.py \
    --servo-port /dev/ttyS1 \
    --lidar-port /dev/ttyUSB0 \
    --cam 0
```

常用参数：`--conf`（检测置信度）、`--safe-dist`（避障距离，单位米）、
`--capture-ratio`（触发捕获的检测框高度比例）。

### 2. ROS 2 方式

```bash
# 编译
colcon build
source install/setup.bash

# 启动完整系统（激光雷达 + 摄像头 + 控制器 + 电机驱动）
ros2 launch rdk_bottle_hunter bottle_hunter.launch.py use_lidar:=true
```

参数配置位于 `src/rdk_bottle_hunter/config/bottle_hunter.yaml`。

#### ROS 2 话题

| 话题                | 类型                              | 说明                     |
| ------------------- | --------------------------------- | ------------------------ |
| `/scan`             | `sensor_msgs/LaserScan`           | VP100 激光雷达扫描       |
| `/bottle_detection` | `rdk_interfaces/BottleDetection`  | 每帧最优瓶子检测结果     |
| `/motor_cmd`        | `rdk_interfaces/MotorCommand`     | 发往 STM32 板的 PWM 指令 |

### 激光雷达可视化

```bash
./start_viz.sh                       # 雷达驱动 + QoS 中继 + rosbridge
python3 lidar_cloud_snapshot.py --frames 5 --out lidar_cloud.png
```

## 许可证

Apache-2.0（详见各功能包清单文件）。
