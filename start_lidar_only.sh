#!/bin/bash
# 仅启动 VP100 激光雷达（无需底盘）
source /opt/ros/humble/setup.bash
source /home/root/RDK_work/install/setup.bash
ros2 launch vp100_ros2 vp100_launch.py
