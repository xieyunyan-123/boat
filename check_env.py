#!/usr/bin/env python3
"""辅助工具：查看 Horizon BPU 运行的软件栈版本和环境信息"""

import subprocess, sys, os, glob
print("=== Board ===")
try:
    with open('/sys/class/socinfo/soc_name') as f:
        print(f"SOC: {f.read().strip()}")
except: pass
print()

print("=== CPU ===")
try:
    subprocess.run(["lscpu"], check=False)
except: pass
print()

print("=== GCC ===")
subprocess.run(["gcc", "--version"], check=False)
print()

print("=== OpenCV ===")
for d in ['/usr/include/opencv4', '/usr/local/include/opencv4']:
    if os.path.isdir(d):
        sp = os.path.join(d, 'opencv2', 'core', 'version.hpp')
        if os.path.isfile(sp):
            with open(sp) as f:
                for l in f:
                    if 'CV_VERSION_' in l and 'STATUS' not in l:
                        print(l.strip())
        break
print()

print("=== Horizon DNN headers ===")
for pattern in ['/usr/include/hobot_dnn/*.h', '/usr/include/dnn/*.h', '/usr/include/hobot_dnn.h']:
    for f in glob.glob(pattern):
        print(f"  {f}")
print()

print("=== Horizon libraries ===")
for pattern in ['/usr/lib/libdnn*', '/usr/lib/libhbrt*', '/usr/lib/libpostprocess*']:
    for f in glob.glob(pattern):
        ls = os.path.getsize(f)
        print(f"  {f}  ({ls} bytes)")
print()

print("=== pkg-config ===")
subprocess.run(["pkg-config", "--list-all", "--cflags", "--libs"], check=False, capture_output=True)
for pkg in ['opencv4', 'opencv', 'dnn']:
    r = subprocess.run(["pkg-config", "--cflags", "--libs", pkg], capture_output=True, text=True)
    print(f"  {pkg}: {r.stdout.strip() if r.returncode == 0 else 'NOT FOUND'}")
