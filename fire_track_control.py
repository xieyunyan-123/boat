#!/usr/bin/env python3
"""
RDK X5 - 火焰检测 + 差速转向控制
=================================
使用 fire_detect.bin + libfire_postprocess.so 检测目标, 根据最大目标在
画面中的水平位置驱动 STM32 舵机板 (复用 stm32_servo.py 的协议)。

控制策略:
    * 未检测到目标        -> CH2 = CH3 = 1500 (中位)
    * 最大目标在中心左侧  -> 减小 CH2/CH3 (向左转)
    * 最大目标在中心右侧  -> 增大 CH2/CH3 (向右转)
    * 对 PWM 输出施加一阶低通滤波(EMA), 使变化平滑

用法:
    python3 fire_track_control.py [--cam 0] [--port /dev/ttyS1]
"""

import numpy as np
import cv2
import ctypes
import json
import time
import os
import argparse

from stm32_servo import STM32ServoController, PWM_MIN, PWM_MAX, PWM_MID

MODEL_PATH = "fire_detect.bin"
LIB_PATH   = "./libfire_postprocess.so"
INPUT_SIZE = 640
CONF_THRES = 0.25
NMS_THRES  = 0.45

CAMERA_ID  = 0
CAM_W      = 640
CAM_H      = 480
CAM_FORMAT = "YUYV"
PAD_COLOR  = 114

# 控制参数
STEER_GAIN   = 200.0   # offset(-1~1) -> PWM 偏移的增益 (us)
DEAD_ZONE    = 0.05    # 中心死区, 归一化偏移小于此值视为居中
FILTER_ALPHA = 0.12    # 低通滤波系数 (越小越平滑, 0~1)
SEND_RATE    = 30.0    # 控制/发送频率 Hz
BASE_SPEED   = 50.0    # 检测到目标时的初速度偏移: CH2=+, CH3=- (us)


# ==================== 加载 C 后处理库 ====================

def _init_pplib(lib_path):
    pp = ctypes.CDLL(lib_path)
    pp.fire_postprocess.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
    ]
    pp.fire_postprocess.restype = ctypes.c_void_p
    pp.fire_postprocess_free.argtypes = [ctypes.c_void_p]
    pp.fire_postprocess_free.restype = None
    return pp


_pp = None


def c_postprocess(outputs, input_size, ori_w, ori_h, conf_thres, nms_thres):
    buffers = [np.array(out.buffer, copy=False) for out in outputs]
    fptrs = [ctypes.cast(buf.ctypes.data, ctypes.POINTER(ctypes.c_float))
             for buf in buffers]

    c_json = _pp.fire_postprocess(
        fptrs[0], fptrs[1], fptrs[2], fptrs[3], fptrs[4], fptrs[5],
        input_size, ori_w, ori_h, conf_thres, nms_thres)

    json_str = ctypes.cast(c_json, ctypes.c_char_p).value.decode()
    _pp.fire_postprocess_free(c_json)

    return json.loads(json_str)


# ==================== 预处理 ====================

def letterbox(img_bgr, target, color=PAD_COLOR):
    h0, w0 = img_bgr.shape[:2]
    r = target / max(h0, w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    if new_w != w0 or new_h != h0:
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = target - new_w, target - new_h
    left, top = dw // 2, dh // 2
    img_bgr = cv2.copyMakeBorder(img_bgr, top, dh - top, left, dw - left,
                                  cv2.BORDER_CONSTANT, value=(color, color, color))
    return img_bgr


def bgr2nv12(img_bgr):
    h, w = img_bgr.shape[:2]
    area = h * w
    yuv420p = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
    nv12 = np.zeros(area * 3 // 2, dtype=np.uint8)
    nv12[:area] = y
    nv12[area:] = uv_packed
    return nv12


# ==================== 目标选择 ====================

def largest_detection(detections):
    """返回面积最大的检测框, 无检测返回 None。"""
    best = None
    best_area = 0.0
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best = det
    return best


# ==================== 控制计算 ====================

def compute_pwm(det, image_w, prev_ch2, prev_ch3):
    """根据最大目标位置计算滤波后的 CH2/CH3。"""
    if det is None:
        target_ch2 = PWM_MID + BASE_SPEED
        target_ch3 = PWM_MID + BASE_SPEED
    else:
        x1, _, x2, _ = det['bbox']
        box_cx = (x1 + x2) / 2.0
        image_cx = image_w / 2.0
        # 归一化偏移: 左侧为负, 右侧为正
        offset = (box_cx - image_cx) / image_cx
        if abs(offset) < DEAD_ZONE:
            offset = 0.0
        # 左侧 -> offset<0 -> delta>0 -> 增大; 右侧 -> offset>0 -> delta<0 -> 减小
        delta = -offset * STEER_GAIN
        # 检测到目标时给一个初速度: CH2=1450, CH3=1550, 再叠加左右转向偏移
        target_ch2 = PWM_MID - BASE_SPEED + delta
        target_ch3 = PWM_MID + BASE_SPEED + delta

    # 一阶低通滤波 (EMA)
    ch2 = FILTER_ALPHA * target_ch2 + (1.0 - FILTER_ALPHA) * prev_ch2
    ch3 = FILTER_ALPHA * target_ch3 + (1.0 - FILTER_ALPHA) * prev_ch3
    return ch2, ch3


# ==================== 主循环 ====================

def main():
    global MODEL_PATH, LIB_PATH, INPUT_SIZE, CAMERA_ID, CAM_W, CAM_H
    global CAM_FORMAT, STEER_GAIN, FILTER_ALPHA, DEAD_ZONE, _pp

    parser = argparse.ArgumentParser(description="Fire detection steering control")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--cam", type=int, default=CAMERA_ID)
    parser.add_argument("--cam-w", type=int, default=CAM_W)
    parser.add_argument("--cam-h", type=int, default=CAM_H)
    parser.add_argument("--cam-format", default=CAM_FORMAT)
    parser.add_argument("--conf", type=float, default=CONF_THRES)
    parser.add_argument("--nms", type=float, default=NMS_THRES)
    parser.add_argument("--port", default="/dev/ttyS1")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--gain", type=float, default=STEER_GAIN)
    parser.add_argument("--alpha", type=float, default=FILTER_ALPHA)
    parser.add_argument("--dead-zone", type=float, default=DEAD_ZONE)
    args = parser.parse_args()

    MODEL_PATH = args.model
    CAMERA_ID = args.cam
    CAM_W = args.cam_w
    CAM_H = args.cam_h
    CAM_FORMAT = args.cam_format
    STEER_GAIN = args.gain
    FILTER_ALPHA = args.alpha
    DEAD_ZONE = args.dead_zone

    workdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workdir)
    LIB_PATH = os.path.join(workdir, "libfire_postprocess.so")
    _pp = _init_pplib(LIB_PATH)

    print(f"[INFO] 加载模型: {MODEL_PATH}")
    from hobot_dnn import pyeasy_dnn as dnn
    models = dnn.load(MODEL_PATH)
    model = models[0]

    inp = model.inputs[0]
    if inp.properties.layout == "NCHW":
        INPUT_SIZE = inp.properties.shape[2]
    else:
        INPUT_SIZE = inp.properties.shape[1]
    print(f"  输入: {INPUT_SIZE}x{INPUT_SIZE}, layout={inp.properties.layout}")

    print(f"[INFO] 打开串口: {args.port} @ {args.baud}")
    ctrl = STM32ServoController(port=args.port, baud=args.baud)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAM_FORMAT))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 /dev/video{CAMERA_ID}")
        ctrl.close()
        return

    print(f"[INFO] 摄像头: {CAM_W}x{CAM_H} @ {CAM_FORMAT}")
    print(f"[INFO] gain={STEER_GAIN} alpha={FILTER_ALPHA} dead_zone={DEAD_ZONE}")
    print(f"[READY] 未检测到目标时 CH2=CH3={PWM_MID}\n")

    ch2_f = float(PWM_MID)
    ch3_f = float(PWM_MID)
    period = 1.0 / SEND_RATE

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_pad = letterbox(frame, INPUT_SIZE)
            nv12_data = bgr2nv12(frame_pad)
            outputs = model.forward(nv12_data)
            detections = c_postprocess(outputs, INPUT_SIZE, CAM_W, CAM_H,
                                       args.conf, args.nms)

            det = largest_detection(detections)
            ch2_f, ch3_f = compute_pwm(det, CAM_W, ch2_f, ch3_f)

            ch2 = max(PWM_MIN, min(PWM_MAX, int(round(ch2_f))))
            ch3 = max(PWM_MIN, min(PWM_MAX, int(round(ch3_f))))
            ctrl.send(PWM_MID, ch3, ch2, PWM_MID, status=1)

            if det is not None:
                x1, _, x2, _ = det['bbox']
                cx = (x1 + x2) / 2.0
                side = "L" if cx < CAM_W / 2 else "R"
                print(f"[TRACK] det={len(detections)} side={side} "
                      f"cx={cx:.0f} CH2={ch2} CH3={ch3} score={det['score']:.2f}",
                      end="\r", flush=True)
            else:
                print(f"[IDLE ] no target        CH2={ch2} CH3={ch3}      ",
                      end="\r", flush=True)

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        print("\n[INFO] 停止, 复位到中位")
    finally:
        try:
            ctrl.send_mid()
            time.sleep(0.05)
        except Exception:
            pass
        ctrl.close()
        cap.release()


if __name__ == "__main__":
    main()
