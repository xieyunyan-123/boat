#!/usr/bin/env python3
"""Analyze LiDAR raw data - try multiple interpretations"""
import serial
import serial.tools.list_ports
import struct

port = None
for p in serial.tools.list_ports.comports():
    if p.vid and p.vid in (0x1A86, 0x10C4, 0x067B, 0x2E3C):
        port = p.device; break
if not port:
    port = '/dev/ttyUSB0'

ser = serial.Serial(port, 230400, timeout=1)
data = ser.read(2000)
ser.close()
print(f"Read {len(data)} bytes\n")

# Find 55 aa sync markers
markers = []
for i in range(len(data) - 1):
    if data[i] == 0x55 and data[i+1] == 0xAA:
        markers.append(i)
print(f"Found {len(markers)} '55 aa' markers at positions: {markers[:20]}")

# Analyze blocks between markers
for idx, start in enumerate(markers[:5]):
    end = markers[idx+1] if idx+1 < len(markers) else min(start + 100, len(data))
    block = data[start:end]
    print(f"\n--- Block {idx} @ offset {start}, {len(block)} bytes ---")
    print(f"Header: {' '.join(f'{b:02x}' for b in block[:8])}")
    payload = block[6:] if len(block) > 6 else block
    # Try: each point = distance(2 LE) + angle(2 LE)
    print("  Trying dist(2) + angle(2):")
    points = []
    for i in range(0, len(payload) - 3, 4):
        d = struct.unpack_from('<H', payload, i)[0]
        a = struct.unpack_from('<H', payload, i+2)[0] * 0.01
        if d > 0 and a <= 360:
            points.append((a, d))
    for a, d in points[:8]:
        print(f"    a={a:7.2f}°  d={d:5}mm ({d/1000:.2f}m)")
    if len(points) > 8:
        print(f"    ... and {len(points)-8} more")
    print(f"  Total valid points: {len(points)}")

    # Try: each point = angle(2 LE) + distance(2 LE)
    print("  Trying angle(2) + dist(2):")
    points2 = []
    for i in range(0, len(payload) - 3, 4):
        a = struct.unpack_from('<H', payload, i)[0] * 0.01
        d = struct.unpack_from('<H', payload, i+2)[0]
        if d > 0 and a <= 360:
            points2.append((a, d))
    for a, d in points2[:8]:
        print(f"    a={a:7.2f}°  d={d:5}mm ({d/1000:.2f}m)")
    if len(points2) > 8:
        print(f"    ... and {len(points2)-8} more")
    print(f"  Total valid points: {len(points2)}")
