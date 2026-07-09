#!/usr/bin/env python3
"""Relay /scan (BEST_EFFORT sensor QoS) to /scan_view (RELIABLE).

rosbridge_server / rviz subscribe with default RELIABLE QoS, which is
incompatible with the vp100 driver's SensorDataQoS publisher. This node
bridges the two so tools like Foxglove Studio can display the scan.
"""

import rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


def main():
    rclpy.init()
    node = rclpy.create_node('scan_relay')

    reliable = QoSProfile(depth=5)
    reliable.reliability = ReliabilityPolicy.RELIABLE
    pub = node.create_publisher(LaserScan, '/scan_view', reliable)

    node.create_subscription(
        LaserScan, '/scan', lambda m: pub.publish(m), qos_profile_sensor_data)

    node.get_logger().info('relaying /scan -> /scan_view (reliable)')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
