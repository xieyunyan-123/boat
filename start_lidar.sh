#!/bin/bash
# 启动 VP100 激光雷达
source /opt/ros/humble/setup.bash
source /home/root/RDK_work/install/setup.bash
ros2 launch originbot_bringup originbot.launch.py use_lidar:=true use_camera:=false
