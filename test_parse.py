#!/usr/bin/env python3
"""Quick test: use full 6-byte sync 55aa070cae59"""
import serial, struct
import serial.tools.list_ports

port = None
for p in serial.tools.list_ports.comports():
    if p.vid and p.vid in (0x1A86, 0x10C4, 0x067B, 0x2E3C): port = p.device; break
if not port: port = '/dev/ttyUSB0'

ser = serial.Serial(port, 230400, timeout=2)
data = ser.read(4000)
ser.close()

SYNC = bytes([0x55, 0xAA, 0x07, 0x0C, 0xAE, 0x59])
pts = []
i = 0
while True:
    idx = data.find(SYNC, i)
    if idx < 0: break
    i = idx + 6
    remaining = len(data) - i
    for j in range(remaining // 4):
        off = i + j * 4
        if off + 3 >= len(data): break
        d = struct.unpack_from('<H', data, off)[0]
        a = struct.unpack_from('<H', data, off + 2)[0] * 0.01
        if a >= 360: a -= 360
        if d > 0 and d < 12000 and 0 <= a < 360:
            pts.append((a, d, j))

print(f"Sync frames found with 6-byte pattern: {data.count(SYNC)}")
print(f"Total points (Dist+Angle): {len(pts)}")
if pts:
    uniq_angles = len(set(round(p[0], 1) for p in pts))
    print(f"Unique angles: {uniq_angles}")
    print(f"Angle range: {min(p[0] for p in pts):.1f}° - {max(p[0] for p in pts):.1f}°")
    print(f"Dist range: {min(p[1] for p in pts)}mm - {max(p[1] for p in pts)}mm")
    # Sample every 100th point
    print(f"Sample every 100th: {[(round(p[0],1),p[1]) for p in pts[::100][:10]]}")
