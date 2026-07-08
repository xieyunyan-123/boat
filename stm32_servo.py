#!/usr/bin/env python3
"""
STM32 Servo UART Communication Module for RDK X5
Protocol: 12-byte frame, 115200 8N1
"""

import serial
import struct
import time
import argparse
import threading


HEADER = b'\xAA\x55'
FRAME_LEN = 12
PWM_MIN = 1000
PWM_MAX = 2000
PWM_MID = 1500


def _clamp_pwm(value):
    return max(PWM_MIN, min(PWM_MAX, value))


def _checksum(data):
    return sum(data[2:11]) & 0xFF


def build_frame(ch0, ch1, ch2, ch3, status=0):
    """Build a 12-byte frame for STM32 servo board."""
    ch0 = _clamp_pwm(ch0)
    ch1 = _clamp_pwm(ch1)
    ch2 = _clamp_pwm(ch2)
    ch3 = _clamp_pwm(ch3)
    buf = bytearray(FRAME_LEN)
    buf[0] = 0xAA
    buf[1] = 0x55
    struct.pack_into('<HHHH', buf, 2, ch0, ch1, ch2, ch3)
    buf[10] = status
    buf[11] = _checksum(buf)
    return bytes(buf)


def parse_frame(data):
    """Parse a 12-byte frame. Returns (ch0, ch1, ch2, ch3, status, sw_mode) or None."""
    if len(data) < FRAME_LEN or data[:2] != HEADER:
        return None
    if _checksum(data[:FRAME_LEN]) != data[11]:
        return None
    ch0, ch1, ch2, ch3 = struct.unpack_from('<HHHH', data, 2)
    return (ch0, ch1, ch2, ch3, data[10])


class STM32ServoController:
    """UART communication controller for STM32 servo board."""

    def __init__(self, port='/dev/ttyS1', baud=115200, timeout=0.01):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.running = False
        self._recv_thread = None
        self._lock = threading.Lock()
        self._last_response = None
        self._response_callback = None

    def send(self, ch0, ch1, ch2, ch3, status=0):
        """Send a single frame (12 bytes) to STM32."""
        frame = build_frame(ch0, ch1, ch2, ch3, status)
        with self._lock:
            self.ser.write(frame)
            self.ser.flush()
        return frame

    def send_mid(self):
        """Send mid-position (1500) to all 4 channels."""
        return self.send(PWM_MID, PWM_MID, PWM_MID, PWM_MID)

    def recv(self, timeout=0.05):
        """Read one response frame from STM32. Returns parsed frame or None."""
        start = time.time()
        buf = bytearray()
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                buf.extend(self.ser.read(self.ser.in_waiting))

            while len(buf) >= FRAME_LEN:
                idx = buf.find(HEADER)
                if idx < 0:
                    buf.clear()
                    break
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < FRAME_LEN:
                    break
                result = parse_frame(bytes(buf[:FRAME_LEN]))
                buf = buf[FRAME_LEN:]
                if result is not None:
                    return result
                buf.pop(0)
            time.sleep(0.001)
        return None

    def _recv_loop(self):
        while self.running:
            response = self.recv(timeout=0.02)
            if response and self._response_callback:
                self._response_callback(response)

    def start_recv(self, callback=None):
        """Start background receive thread."""
        self._response_callback = callback
        self.running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def stop_recv(self):
        """Stop background receive thread."""
        self.running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=1.0)

    def close(self):
        """Close serial port."""
        self.stop_recv()
        self.ser.close()


def main():
    parser = argparse.ArgumentParser(description='RDK X5 STM32 Servo UART Controller')
    parser.add_argument('--port', '-p', default='/dev/ttyS1', help='Serial port')
    parser.add_argument('--baud', '-b', type=int, default=115200, help='Baud rate')
    parser.add_argument('--interval', '-i', type=float, default=0.1,
                        help='Send interval in seconds')
    parser.add_argument('--ch1', type=int, default=1500, help='CH1 PWM (1000-2000)')
    parser.add_argument('--ch2', type=int, default=1500, help='CH2 PWM (1000-2000)')
    parser.add_argument('--ch3', type=int, default=1500, help='CH3 PWM (1000-2000)')
    parser.add_argument('--ch4', type=int, default=1500, help='CH4 PWM (1000-2000)')
    parser.add_argument('--once', '-o', action='store_true', help='Send once and exit')
    parser.add_argument('--mid', '-m', action='store_true', help='Send mid-position (1500) and exit')

    args = parser.parse_args()

    ctrl = STM32ServoController(port=args.port, baud=args.baud)
    print(f"[INFO] {args.port} @ {args.baud} baud")
    print(f"[INFO] PWM range: {PWM_MIN}-{PWM_MAX} us, mid: {PWM_MID} us")

    def _on_response(data):
        ch0, ch1, ch2, ch3, status = data
        mode = 'UART' if status == 1 else 'PASSTHROUGH'
        print(f"[RECV] CH1={ch0} CH2={ch1} CH3={ch2} CH4={ch3} mode={mode}")

    ctrl.start_recv(callback=_on_response)

    try:
        if args.mid:
            frame = ctrl.send_mid()
            print(f"[SEND] MID position to all channels")
            print(f"       {' '.join(f'{b:02X}' for b in frame)}")
            time.sleep(0.1)
        elif args.once:
            frame = ctrl.send(args.ch1, args.ch2, args.ch3, args.ch4)
            print(f"[SEND] CH1={args.ch1} CH2={args.ch2} CH3={args.ch3} CH4={args.ch4}")
            print(f"       {' '.join(f'{b:02X}' for b in frame)}")
            time.sleep(0.1)
        else:
            print(f"[INFO] Sending every {args.interval*1000:.0f}ms")
            print(f"       CH1={args.ch1} CH2={args.ch2} CH3={args.ch3} CH4={args.ch4}")
            while True:
                frame = ctrl.send(args.ch1, args.ch2, args.ch3, args.ch4)
                hex_str = ' '.join(f'{b:02X}' for b in frame)
                print(f"[SEND] {hex_str}")
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped")
    finally:
        ctrl.close()


if __name__ == '__main__':
    main()
