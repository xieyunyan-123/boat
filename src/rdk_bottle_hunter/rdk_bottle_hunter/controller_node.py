#!/usr/bin/env python3
"""High-level state-machine controller.

Fuses bottle detections (``bottle_detection``) and LiDAR scans (``scan``)
and publishes motor commands (``motor_cmd``). Behaviour mirrors the original
monolithic ``main.py``:

  SEARCH   - no target: hold still and wait
  APPROACH - bottle visible: differential steering toward it
  CAPTURE  - bottle very close: spin-forward grab for a fixed duration
  AVOID    - wall/obstacle ahead: fixed turn manoeuvre
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from rdk_interfaces.msg import BottleDetection, MotorCommand
from rdk_bottle_hunter.stm32_protocol import PWM_MID, PWM_MIN, PWM_MAX

STATE_SEARCH = 0
STATE_APPROACH = 1
STATE_CAPTURE = 2
STATE_AVOID = 3
STATE_NAMES = {0: 'SEARCH', 1: 'APPROACH', 2: 'CAPTURE', 3: 'AVOID'}


class ControllerNode(Node):

    def __init__(self):
        super().__init__('controller_node')

        # Motion tuning
        self.declare_parameter('base_speed', 200)
        self.declare_parameter('diff_max', 200)
        # Capture behaviour
        self.declare_parameter('capture_bbox_ratio', 0.7)
        self.declare_parameter('capture_near_frames', 3)
        self.declare_parameter('capture_ch2', 1800)
        self.declare_parameter('capture_ch3', 1200)
        self.declare_parameter('capture_duration', 1.0)
        # Obstacle / avoidance
        self.declare_parameter('wall_min_dist', 0.5)
        self.declare_parameter('forward_sector_deg', 30.0)
        self.declare_parameter('obstacle_min_pts', 3)
        self.declare_parameter('avoid_ch2', 1800)
        self.declare_parameter('avoid_ch3', 1200)
        self.declare_parameter('avoid_duration', 2.0)
        # Detection staleness
        self.declare_parameter('detection_timeout', 0.5)
        self.declare_parameter('control_rate', 20.0)

        g = self.get_parameter
        self.base_speed = g('base_speed').value
        self.diff_max = g('diff_max').value
        self.center_speed = PWM_MID + self.base_speed
        self.capture_bbox_ratio = g('capture_bbox_ratio').value
        self.capture_near_frames = g('capture_near_frames').value
        self.capture_ch2 = g('capture_ch2').value
        self.capture_ch3 = g('capture_ch3').value
        self.capture_duration = g('capture_duration').value
        self.wall_min_dist = g('wall_min_dist').value
        self.forward_sector = math.radians(g('forward_sector_deg').value)
        self.obstacle_min_pts = g('obstacle_min_pts').value
        self.avoid_ch2 = g('avoid_ch2').value
        self.avoid_ch3 = g('avoid_ch3').value
        self.avoid_duration = g('avoid_duration').value
        self.detection_timeout = g('detection_timeout').value
        control_rate = g('control_rate').value

        self.state = STATE_SEARCH
        self.state_start = self.now()
        self.near_streak = 0

        self.last_detection = None
        self.last_detection_time = 0.0
        self.last_scan = None

        self.cmd_pub = self.create_publisher(MotorCommand, 'motor_cmd', 10)
        self.create_subscription(BottleDetection, 'bottle_detection',
                                 self.on_detection, 10)
        self.create_subscription(LaserScan, 'scan',
                                 self.on_scan, qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / control_rate, self.on_control)

        self.get_logger().info(
            f'Controller ready (safe_dist={self.wall_min_dist}m, '
            f'capture_ratio={self.capture_bbox_ratio:.0%})')

    # ---- helpers ----------------------------------------------------------

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_cmd(self, ch1, ch2, ch3, ch4, status=0):
        msg = MotorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ch1, msg.ch2, msg.ch3, msg.ch4 = (int(ch1), int(ch2), int(ch3), int(ch4))
        msg.status = status
        self.cmd_pub.publish(msg)

    def set_state(self, new_state):
        if new_state != self.state:
            self.get_logger().info(
                f'{STATE_NAMES[self.state]} -> {STATE_NAMES[new_state]}')
        self.state = new_state
        self.state_start = self.now()

    # ---- callbacks --------------------------------------------------------

    def on_detection(self, msg: BottleDetection):
        self.last_detection = msg
        self.last_detection_time = self.now()

    def on_scan(self, msg: LaserScan):
        self.last_scan = msg

    def detect_obstacle(self):
        """Return (is_obstacle, nearest_dist, near_count) from the last scan."""
        scan = self.last_scan
        if scan is None:
            return False, float('inf'), 0
        near = 0
        min_d = float('inf')
        angle = scan.angle_min
        for r in scan.ranges:
            a = angle
            angle += scan.angle_increment
            if math.isinf(r) or math.isnan(r) or r <= 0.0:
                continue
            # normalise angle to [-pi, pi] and keep the forward sector
            a_norm = math.atan2(math.sin(a), math.cos(a))
            if abs(a_norm) <= self.forward_sector:
                if r < min_d:
                    min_d = r
                if r < self.wall_min_dist:
                    near += 1
        is_obstacle = near >= self.obstacle_min_pts and min_d < self.wall_min_dist
        return is_obstacle, min_d, near

    def compute_approach(self, det: BottleDetection):
        """Return (ch1, ch2, ch3, ch4, target_state) for a visible bottle."""
        box_cx = (det.x_min + det.x_max) / 2.0
        box_h = det.y_max - det.y_min
        image_cx = det.image_width / 2.0
        offset = (box_cx - image_cx) / image_cx
        ratio = box_h / float(det.image_height)

        if ratio >= self.capture_bbox_ratio:
            return (PWM_MID, self.capture_ch2, self.capture_ch3, PWM_MID,
                    STATE_CAPTURE)

        ch2 = int(self.center_speed - offset * (self.diff_max / 2))
        ch3 = int(self.center_speed + offset * (self.diff_max / 2))
        ch2 = max(PWM_MIN, min(PWM_MAX, ch2))
        ch3 = max(PWM_MIN, min(PWM_MAX, ch3))
        return PWM_MID, ch2, ch3, PWM_MID, STATE_APPROACH

    # ---- main loop --------------------------------------------------------

    def on_control(self):
        t = self.now()
        is_obstacle, min_dist, near_pts = self.detect_obstacle()

        det = self.last_detection
        if det is not None and (t - self.last_detection_time) > self.detection_timeout:
            det = None
        has_bottle = det is not None and det.detected

        # 1) Obstacle overrides everything except an in-progress capture.
        if is_obstacle and self.state not in (STATE_AVOID, STATE_CAPTURE):
            self.set_state(STATE_AVOID)
            self.get_logger().warn(
                f'obstacle! dist={min_dist:.2f}m pts={near_pts}')
            self.publish_cmd(PWM_MID, self.avoid_ch2, self.avoid_ch3, PWM_MID)
            return

        # 2) Timed states run to completion.
        if self.state == STATE_AVOID:
            if t - self.state_start >= self.avoid_duration:
                self.set_state(STATE_SEARCH)
                self.publish_cmd(PWM_MID, PWM_MID, PWM_MID, PWM_MID)
            return

        if self.state == STATE_CAPTURE:
            if t - self.state_start >= self.capture_duration:
                self.set_state(STATE_SEARCH)
                self.publish_cmd(PWM_MID, PWM_MID, PWM_MID, PWM_MID)
            return

        # 3) SEARCH / APPROACH.
        if has_bottle:
            ch1, ch2, ch3, ch4, target = self.compute_approach(det)
            if target == STATE_CAPTURE:
                self.near_streak += 1
                if self.near_streak >= self.capture_near_frames:
                    self.set_state(STATE_CAPTURE)
                    self.publish_cmd(ch1, ch2, ch3, ch4)
                    self.near_streak = 0
                else:
                    self.publish_cmd(PWM_MID, PWM_MID, PWM_MID, PWM_MID)
            else:
                self.near_streak = 0
                self.set_state(STATE_APPROACH)
                self.publish_cmd(ch1, ch2, ch3, ch4)
        else:
            self.near_streak = 0
            self.set_state(STATE_SEARCH)
            self.publish_cmd(PWM_MID, PWM_MID, PWM_MID, PWM_MID)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
