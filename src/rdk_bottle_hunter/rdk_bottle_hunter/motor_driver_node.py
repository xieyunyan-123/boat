#!/usr/bin/env python3
"""STM32 motor/servo driver node.

Subscribes to ``motor_cmd`` (rdk_interfaces/MotorCommand) and forwards each
command as a 12-byte UART frame to the STM32 servo board. On shutdown (and
when no command has been received within ``watchdog_timeout`` seconds) it
sends the neutral/mid position so the robot stops safely.
"""

import serial

import rclpy
from rclpy.node import Node

from rdk_interfaces.msg import MotorCommand
from rdk_bottle_hunter.stm32_protocol import build_frame, PWM_MID


class MotorDriverNode(Node):

    def __init__(self):
        super().__init__('motor_driver_node')

        self.declare_parameter('port', '/dev/ttyS1')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('watchdog_timeout', 1.0)

        self.port = self.get_parameter('port').value
        self.baud = self.get_parameter('baud').value
        self.watchdog_timeout = self.get_parameter('watchdog_timeout').value

        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
            self.get_logger().info(f'STM32 connected: {self.port} @ {self.baud}')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Cannot open {self.port}: {exc}')
            raise

        self._last_cmd_time = self.get_clock().now()
        self.send_mid()

        self.sub = self.create_subscription(
            MotorCommand, 'motor_cmd', self.on_cmd, 10)
        self.watchdog = self.create_timer(0.2, self.on_watchdog)

    def _write(self, ch1, ch2, ch3, ch4, status=0):
        try:
            self.ser.write(build_frame(ch1, ch2, ch3, ch4, status))
            self.ser.flush()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'UART write failed: {exc}')

    def send_mid(self):
        self._write(PWM_MID, PWM_MID, PWM_MID, PWM_MID)

    def on_cmd(self, msg: MotorCommand):
        self._last_cmd_time = self.get_clock().now()
        self._write(msg.ch1, msg.ch2, msg.ch3, msg.ch4, msg.status)

    def on_watchdog(self):
        dt = (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9
        if dt > self.watchdog_timeout:
            self.send_mid()

    def destroy_node(self):
        try:
            self.send_mid()
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MotorDriverNode()
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
