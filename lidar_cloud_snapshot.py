#!/usr/bin/env python3
"""Capture LiDAR /scan and render a top-down point cloud to a PNG.

Headless-friendly (matplotlib Agg backend). Subscribes to /scan with sensor
QoS, accumulates a few sweeps, converts polar -> cartesian and saves an image.

Usage:
    python3 lidar_cloud_snapshot.py [--frames 5] [--out lidar_cloud.png]
"""

import argparse
import math
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=5, help='sweeps to accumulate')
    ap.add_argument('--out', default='lidar_cloud.png')
    ap.add_argument('--timeout', type=float, default=10.0)
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node('lidar_cloud_snapshot')

    frames = []

    def cb(msg):
        frames.append(msg)

    node.create_subscription(LaserScan, '/scan', cb, qos_profile_sensor_data)

    print(f'[INFO] waiting for {args.frames} scans on /scan ...')
    t0 = time.time()
    while rclpy.ok() and len(frames) < args.frames and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)

    node.destroy_node()
    rclpy.shutdown()

    if not frames:
        print('[ERROR] no /scan received. Is the lidar node running?')
        return 1

    xs, ys, dists = [], [], []
    for m in frames:
        a = m.angle_min
        for r in m.ranges:
            ang = a
            a += m.angle_increment
            if math.isinf(r) or math.isnan(r) or r <= 0.0:
                continue
            xs.append(r * math.cos(ang))
            ys.append(r * math.sin(ang))
            dists.append(r)

    last = frames[-1]
    rng = last.range_max if last.range_max < 20 else 5.0
    print(f'[INFO] {len(frames)} sweeps, {len(xs)} valid points, '
          f'frame_id={last.header.frame_id}')

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(xs, ys, c=dists, cmap='viridis', s=6)
    ax.scatter([0], [0], c='red', marker='^', s=120, label='lidar')
    ax.arrow(0, 0, rng * 0.25, 0, head_width=rng * 0.03,
             color='red', alpha=0.6)
    ax.text(rng * 0.28, 0, 'front (x+)', color='red', va='center')
    ax.set_aspect('equal')
    ax.set_xlabel('X (m, forward)')
    ax.set_ylabel('Y (m, left)')
    ax.set_title(f'LiDAR point cloud  ({len(xs)} pts, {len(frames)} sweeps)')
    ax.grid(True, alpha=0.3)
    lim = max(1.0, min(rng, max((max(map(abs, xs), default=1),
                                 max(map(abs, ys), default=1))) * 1.1))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.colorbar(sc, ax=ax, label='range (m)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f'[OK] saved {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
