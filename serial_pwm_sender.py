#!/usr/bin/env python3
"""
Serial PWM Data Sender for RDK X5 (Binary Protocol)
Packet: 2B header + 16B data + 1B status + 1B checksum = 20 bytes
"""

import serial
import struct
import time
import argparse


HEADER = b'\xAA\x55'

STATUS_LABEL = {0: 'STOP', 1: 'RUN', 2: 'ERROR'}


def build_packet(ch1, ch2, ch3, ch4, status):
    payload = struct.pack('<IIIIB', ch1, ch2, ch3, ch4, status)
    checksum = 0
    for b in payload:
        checksum ^= b
    return HEADER + payload + bytes([checksum & 0xFF])


def parse_packet(data):
    if len(data) < 20 or data[:2] != HEADER:
        return None
    payload = data[2:19]
    checksum = 0
    for b in payload:
        checksum ^= b
    if checksum & 0xFF != data[19]:
        return None
    return struct.unpack('<IIIIB', payload)


def main():
    parser = argparse.ArgumentParser(description='RDK X5 Serial PWM Sender (Binary)')
    parser.add_argument('--port', '-p', default='/dev/ttyS1')
    parser.add_argument('--baud', '-b', type=int, default=115200)
    parser.add_argument('--interval', '-i', type=float, default=1.0)
    parser.add_argument('--ch1', type=int, default=1500)
    parser.add_argument('--ch2', type=int, default=1500)
    parser.add_argument('--ch3', type=int, default=1500)
    parser.add_argument('--ch4', type=int, default=1500)
    parser.add_argument('--status', '-s', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--once', '-o', action='store_true')

    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"[INFO] {args.port} @ {args.baud} baud")

    try:
        while True:
            pkt = build_packet(args.ch1, args.ch2, args.ch3, args.ch4, args.status)
            ser.write(pkt)
            ser.flush()

            hex_str = ' '.join(f'{b:02X}' for b in pkt)
            print(f"[SEND] CH1={args.ch1} CH2={args.ch2} CH3={args.ch3} "
                  f"CH4={args.ch4} STATUS={STATUS_LABEL[args.status]}")
            print(f"       {hex_str}")

            if args.once:
                break
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped")
    finally:
        ser.close()


if __name__ == '__main__':
    main()
