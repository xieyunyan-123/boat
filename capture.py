#!/usr/bin/env python3

import cv2
import os
from datetime import datetime

SAVE_DIR = "/home/root/captures"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera: {w}x{h}")
print("Press ENTER to capture | 'q' + ENTER to quit")
print(f"Saving to: {SAVE_DIR}")

count = 0
while True:
    cmd = input("> ")
    if cmd.strip().lower() == "q":
        break

    for _ in range(5):
        cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        print("Error: Failed to capture frame")
        continue

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"capture_{ts}.png"
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, frame)
    count += 1
    print(f"Saved: {filename}  (total: {count})")

cap.release()
print(f"Done. {count} photos saved to {SAVE_DIR}")
