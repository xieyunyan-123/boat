#!/usr/bin/env bash
# Start the live LiDAR visualization stack (lidar driver + QoS relay + rosbridge).
# Everything is detached; logs go to /tmp. Stop with stop_viz.sh.
source /opt/ros/humble/setup.bash
source /home/root/RDK_work/install/setup.bash

pkill -9 -f vp100_ros2_node 2>/dev/null
pkill -9 -f scan_relay 2>/dev/null
pkill -9 -f rosbridge_websocket 2>/dev/null
sleep 1

ros2 run vp100_ros2 vp100_ros2_node --ros-args \
    --params-file /home/root/RDK_work/src/vp100_ros2/params/vp100.yaml \
    >/tmp/vp100.log 2>&1 &

python3 /home/root/RDK_work/scan_relay.py >/tmp/relay.log 2>&1 &

ros2 run rosbridge_server rosbridge_websocket >/tmp/rosbridge.log 2>&1 &

wait
