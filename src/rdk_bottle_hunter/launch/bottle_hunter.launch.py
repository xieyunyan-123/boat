#!/usr/bin/env python3
"""Bring up the full bottle-hunter stack.

Launches (in one process group):
  * vp100_ros2 LiDAR driver (publishes /scan)   -- via its own launch file
  * camera_detector_node  (camera + YOLOv5 BPU) -> /bottle_detection
  * motor_driver_node     (/motor_cmd -> STM32 UART)
  * controller_node       (state machine: /bottle_detection + /scan -> /motor_cmd)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('rdk_bottle_hunter')
    params_file = os.path.join(pkg_share, 'config', 'bottle_hunter.yaml')

    use_lidar = LaunchConfiguration('use_lidar')

    vp100_share = get_package_share_directory('vp100_ros2')
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(vp100_share, 'launch', 'vp100_launch.py')),
        condition=IfCondition(use_lidar),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_lidar', default_value='true',
                              description='Also launch the vp100 LiDAR driver.'),
        DeclareLaunchArgument('params_file', default_value=params_file,
                              description='Path to the node parameters YAML.'),

        lidar_launch,

        Node(package='rdk_bottle_hunter', executable='camera_detector_node',
             name='camera_detector_node', output='screen',
             parameters=[LaunchConfiguration('params_file')]),

        Node(package='rdk_bottle_hunter', executable='motor_driver_node',
             name='motor_driver_node', output='screen',
             parameters=[LaunchConfiguration('params_file')]),

        Node(package='rdk_bottle_hunter', executable='controller_node',
             name='controller_node', output='screen',
             parameters=[LaunchConfiguration('params_file')]),
    ])
