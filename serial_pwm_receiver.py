#!/usr/bin/env python3
"""
Serial PWM Data Packet Receiver (for verifying with CH340)
Run this on the PC side connected via CH340 USB-to-TTL.
"""

import serial
import struct
import argparse
import time


HEADER = b'\xAA\x55'


def main():
    parser = argparse.ArgumentParser(description='PWM Packet Receiver')
    parser.add_argument('--port', '-p', required=True, help='Serial port (e.g. COM3 or /dev/ttyUSB0)')
    parser.add_argument('--baud', '-b', type=int, default=115200, help='Baud rate')
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"[INFO] Listening on {args.port} at {args.baud} baud")
    print("[INFO] Waiting for PWM packets...")

    buf = bytearray()
    try:
        while True:
            if ser.in_waiting:
                buf.extend(ser.read(ser.in_waiting))

            while len(buf) >= 20:
                idx = buf.find(HEADER)
                if idx < 0:
                    buf.clear()
                    break

                if idx > 0:
                    buf = buf[idx:]

                if len(buf) < 20:
                    break

                raw = bytes(buf[:20])
                payload = raw[2:19]
                checksum = raw[19]

                xor_sum = 0
                for b in payload:
                    xor_sum ^= b
                if xor_sum & 0xFF != checksum:
                    buf.pop(0)
                    continue

                ch1, ch2, ch3, ch4, status = struct.unpack('<IIIIB', payload)
                print(f"[RECV] CH1={ch1}Hz CH2={ch2}Hz CH3={ch3}Hz CH4={ch4}Hz Status={status}")
                print(f"       HEX: {' '.join(f'{b:02X}' for b in raw)}")
                buf = buf[20:]

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped")
    finally:
        ser.close()


if __name__ == '__main__':
    main()
