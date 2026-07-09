#!/usr/bin/env python3
"""
RDK X5 Main Controller - 总指挥
================================
YOLOv5 瓶子检测 + LiDAR 雷达避障 + STM32 Servo 控制

功能:
  1. YOLOv5 实时检测摄像头中的瓶子位置
  2. 根据瓶子在画面中的偏移量控制电机:
     - 瓶子在左侧  -> 左前方 (CH2较大, CH3较小, 差速左转前进)
     - 瓶子在右侧  -> 右前方 (CH3较大, CH2较小, 差速右转前进)
     - 瓶子在中间  -> 直行 (CH2=CH3)
     - 瓶子消失(即将碰撞) -> 旋转前进捕获 (CH2正转, CH3反转)
  3. LiDAR 雷达检测前方线状障碍物(墙壁), 发现后自动180度转向避障

用法:
    python3 main.py [--servo-port /dev/ttyS1] [--lidar-port /dev/ttyUSB0] [--cam N]

依赖模块:
    stm32_servo.py  (STM32 舵机 UART 通信)
    hobot_dnn       (地平线 BPU 推理库)
    libpostprocess.so (C++ YOLOv5 后处理)
"""

import numpy as np
import cv2
import ctypes
import json
import time
import threading
import struct
import os
import sys
import signal
import argparse

import serial
import serial.tools.list_ports

from stm32_servo import STM32ServoController, PWM_MID, PWM_MIN, PWM_MAX, build_frame

# ==================== 配置参数 ====================

MODEL_PATH       = "best_detect_640x640_bayese_nv12.bin"
INPUT_SIZE       = 640
NUM_CLASSES      = 1
CONF_THRES       = 0.4
NMS_THRES        = 0.45
NMS_TOP_K        = 500
IS_PAD_RESIZE    = 1
PAD_COLOR        = 114

CAMERA_ID        = 0
CAM_W            = 640
CAM_H            = 480

SERVO_PORT       = "/dev/ttyS1"
SERVO_BAUD       = 115200

LIDAR_PORT       = "/dev/ttyUSB0"
LIDAR_BAUD       = 230400

BASE_SPEED       = 200
DIFF_MAX         = 200
CENTER_SPEED     = PWM_MID + BASE_SPEED

CAPTURE_BBOX_RATIO   = 0.7
CAPTURE_NEAR_FRAMES  = 3
CAPTURE_CH2     = 1800
CAPTURE_CH3     = 1200
CAPTURE_DURATION     = 1.0

WALL_MIN_DIST    = 0.5
FORWARD_SECTOR   = 30
OBSTACLE_MIN_PTS = 3
AVOID_CH2        = 1800
AVOID_CH3        = 1200
AVOID_DURATION   = 2.0

STATE_SEARCH   = 0
STATE_APPROACH = 1
STATE_CAPTURE  = 2
STATE_AVOID    = 3

# ==================== C 结构体 (YOLOv5 后处理) ====================

class hbSysMem_t(ctypes.Structure):
    _fields_ = [
        ("phyAddr", ctypes.c_double),
        ("virAddr", ctypes.c_void_p),
        ("memSize", ctypes.c_int),
    ]

class hbDNNQuantiShift_yt(ctypes.Structure):
    _fields_ = [
        ("shiftLen", ctypes.c_int),
        ("shiftData", ctypes.c_char_p),
    ]

class hbDNNQuantiScale_t(ctypes.Structure):
    _fields_ = [
        ("scaleLen", ctypes.c_int),
        ("scaleData", ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen", ctypes.c_int),
        ("zeroPointData", ctypes.c_char_p),
    ]

class hbDNNTensorShape_t(ctypes.Structure):
    _fields_ = [
        ("dimensionSize", ctypes.c_int * 8),
        ("numDimensions", ctypes.c_int),
    ]

class hbDNNTensorProperties_t(ctypes.Structure):
    _fields_ = [
        ("validShape", hbDNNTensorShape_t),
        ("alignedShape", hbDNNTensorShape_t),
        ("tensorLayout", ctypes.c_int),
        ("tensorType", ctypes.c_int),
        ("shift", hbDNNQuantiShift_yt),
        ("scale", hbDNNQuantiScale_t),
        ("quantiType", ctypes.c_int),
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize", ctypes.c_int),
        ("stride", ctypes.c_int * 8),
    ]

class hbDNNTensor_t(ctypes.Structure):
    _fields_ = [
        ("sysMem", hbSysMem_t * 4),
        ("properties", hbDNNTensorProperties_t),
    ]

class Yolov5PostProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("height", ctypes.c_int),
        ("width", ctypes.c_int),
        ("ori_height", ctypes.c_int),
        ("ori_width", ctypes.c_int),
        ("score_threshold", ctypes.c_float),
        ("nms_threshold", ctypes.c_float),
        ("nms_top_k", ctypes.c_int),
        ("is_pad_resize", ctypes.c_int),
    ]

# ==================== YOLOv5 工具函数 ====================

def letterbox(img_bgr, target=INPUT_SIZE, color=PAD_COLOR):
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


def get_tensor_layout(layout_str):
    return 2 if layout_str == "NCHW" else 0


def init_postprocess_lib():
    lib = ctypes.CDLL('/usr/lib/libpostprocess.so')
    lib.Yolov5PostProcess.argtypes = [ctypes.POINTER(Yolov5PostProcessInfo_t)]
    lib.Yolov5PostProcess.restype = ctypes.c_char_p
    return lib


def setup_output_tensors(model, lib):
    out_count = len(model.outputs)
    tensors = (hbDNNTensor_t * out_count)()
    for i in range(out_count):
        out_prop = model.outputs[i].properties
        tensors[i].properties.tensorLayout = get_tensor_layout(out_prop.layout)
        has_scale = len(out_prop.scale_data) > 0
        if has_scale:
            tensors[i].properties.quantiType = 2
            scale_data = out_prop.scale_data
            tensors[i].properties.scale.scaleData = \
                scale_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            tensors[i].properties.quantiType = 0
        shape = out_prop.shape
        for j in range(len(shape)):
            tensors[i].properties.validShape.dimensionSize[j] = shape[j]
            tensors[i].properties.alignedShape.dimensionSize[j] = shape[j]
    return tensors


def run_postprocess(lib, outputs, output_tensors, post_info):
    out_count = len(outputs)
    for i in range(out_count):
        if output_tensors[i].properties.quantiType == 0:
            ptr = ctypes.cast(
                outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_void_p)
        else:
            ptr = ctypes.cast(
                outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_void_p)
        output_tensors[i].sysMem[0].virAddr = ptr
        lib.Yolov5doProcess(output_tensors[i], ctypes.pointer(post_info), i)

    result_bytes = lib.Yolov5PostProcess(ctypes.pointer(post_info))
    if not result_bytes:
        return []
    result_str = result_bytes.decode('utf-8', errors='replace')
    idx = result_str.find('[')
    if idx < 0:
        return []
    try:
        return json.loads(result_str[idx:])
    except json.JSONDecodeError:
        return []


def load_yolov5_model(model_path):
    print(f"[INFO] 加载模型: {model_path}")
    try:
        from hobot_dnn import pyeasy_dnn as dnn
    except ImportError as e:
        print(f"[ERROR] 无法导入 hobot_dnn: {e}")
        sys.exit(1)
    models = dnn.load(model_path)
    model = models[0]

    inp = model.inputs[0]
    if inp.properties.layout == "NCHW":
        inp_size = inp.properties.shape[2]
    else:
        inp_size = inp.properties.shape[1]
    print(f"  输入: {inp_size}x{inp_size}, 输出层: {len(model.outputs)}")

    na = 3
    num_cls = model.outputs[0].properties.shape[-1] // na - 5
    print(f"  推导类别数: {num_cls}")

    return model, inp_size, num_cls


# ==================== LiDAR ====================

lidar_latest_points = []
lidar_lock = threading.Lock()


def find_lidar_port():
    for p in serial.tools.list_ports.comports():
        if p.vid and p.vid in (0x1A86, 0x10C4, 0x067B, 0x2E3C):
            return p.device
    for p in serial.tools.list_ports.comports():
        if 'USB' in p.description.upper() or 'usb' in p.device.lower():
            return p.device
    return '/dev/ttyUSB0'


def parse_lidar_points(data):
    points = []
    i = 0
    while i < len(data) - 5:
        if data[i] == 0x55 and data[i+1] == 0xAA:
            hdr = 6
            remaining = len(data) - i - hdr
            n_points = remaining // 4
            for j in range(n_points):
                offset = i + hdr + j * 4
                if offset + 3 >= len(data):
                    break
                a_raw = struct.unpack_from('<H', data, offset)[0]
                d_raw = struct.unpack_from('<H', data, offset + 2)[0]
                a = a_raw * 0.01
                if a >= 360: a -= 360.0
                if a < 0: a += 360.0
                if d_raw > 0 and d_raw < 12000 and 0 <= a < 360:
                    points.append({"a": round(a, 2), "d": round(d_raw / 1000.0, 3)})
            i += hdr
        else:
            i += 1
    return points


def lidar_reader_loop(port, baud):
    global lidar_latest_points
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except Exception as e:
        print(f"[ERROR] 无法打开 LiDAR 端口 {port}: {e}")
        return
    buf = bytearray()
    print(f"[INFO] LiDAR 已连接: {port} @ {baud}")
    while True:
        try:
            if ser.in_waiting:
                buf.extend(ser.read(ser.in_waiting))
        except Exception:
            time.sleep(0.1)
            continue
        if len(buf) > 8192:
            buf = buf[-4096:]
        pts = parse_lidar_points(buf)
        if pts:
            with lidar_lock:
                lidar_latest_points = pts


def detect_obstacle(points, min_dist=WALL_MIN_DIST, sector=FORWARD_SECTOR, min_pts=OBSTACLE_MIN_PTS):
    """检测前方是否存在墙壁/障碍物。

    判定条件:
      在前方 ±sector 度扇区内，有 >= min_pts 个点距离 < min_dist 米。
      这些点若分布在靠近的连续角度上，则判定为线状障碍（墙壁）。

    Returns:
        (is_obstacle: bool, nearest_dist: float, near_count: int)
    """
    count_near = 0
    min_d = float('inf')

    for p in points:
        a = p['a']
        d = p['d']
        a_norm = a if a <= 180 else a - 360
        if abs(a_norm) <= sector:
            if d < min_d:
                min_d = d
            if d < min_dist:
                count_near += 1

    is_obstacle = count_near >= min_pts and min_d < min_dist
    return is_obstacle, min_d, count_near


# ==================== 电机控制 ====================

def compute_motors(bottle_left, bottle_top, bottle_right, bottle_bottom,
                   image_w, image_h):
    """根据瓶子 bbox 计算 CH2/CH3 电机值。

    Returns:
        (ch1, ch2, ch3, ch4, state)
    """
    if bottle_left is None:
        return PWM_MID, PWM_MID, PWM_MID, PWM_MID, STATE_SEARCH

    box_cx = (bottle_left + bottle_right) / 2.0
    box_h  = bottle_bottom - bottle_top

    image_cx = image_w / 2.0
    offset = (box_cx - image_cx) / (image_cx)

    ratio = box_h / float(image_h)

    if ratio >= CAPTURE_BBOX_RATIO:
        return PWM_MID, CAPTURE_CH2, CAPTURE_CH3, PWM_MID, STATE_CAPTURE

    ch2 = int(CENTER_SPEED - offset * (DIFF_MAX / 2))
    ch3 = int(CENTER_SPEED + offset * (DIFF_MAX / 2))

    ch2 = max(PWM_MIN, min(PWM_MAX, ch2))
    ch3 = max(PWM_MIN, min(PWM_MAX, ch3))

    return PWM_MID, ch2, ch3, PWM_MID, STATE_APPROACH


# ==================== 状态机 ====================

class MainController:
    def __init__(self, servo_port=SERVO_PORT, servo_baud=SERVO_BAUD,
                 lidar_port=LIDAR_PORT, lidar_baud=LIDAR_BAUD,
                 camera_id=CAMERA_ID, model_path=MODEL_PATH):
        self.servo = None
        self.servo_port = servo_port
        self.servo_baud = servo_baud
        self.lidar_port = lidar_port
        self.lidar_baud = lidar_baud
        self.camera_id = camera_id
        self.model_path = model_path

        self.state = STATE_SEARCH
        self.state_start = 0.0
        self.capture_triggered = False
        self.near_count = 0
        self.running = True

        self.cap = None
        self.model = None
        self.lib = None
        self.output_tensors = None
        self.post_info = None

        self.detections_log = []

    def setup(self):
        if not os.path.exists(self.model_path):
            alt = "/app/best_detect_640x640_bayese_nv12.bin"
            if os.path.exists(alt):
                self.model_path = alt
                print(f"[INFO] 使用备用模型路径: {alt}")
            else:
                print(f"[ERROR] 找不到模型文件: {self.model_path}")
                sys.exit(1)

        self.model, input_size, _ = load_yolov5_model(self.model_path)
        global INPUT_SIZE
        INPUT_SIZE = input_size

        self.lib = init_postprocess_lib()
        self.output_tensors = setup_output_tensors(self.model, self.lib)

        self.post_info = Yolov5PostProcessInfo_t()
        self.post_info.height = INPUT_SIZE
        self.post_info.width = INPUT_SIZE
        self.post_info.ori_height = CAM_H
        self.post_info.ori_width = CAM_W
        self.post_info.score_threshold = CONF_THRES
        self.post_info.nms_threshold = NMS_THRES
        self.post_info.nms_top_k = NMS_TOP_K
        self.post_info.is_pad_resize = IS_PAD_RESIZE

        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print(f"[ERROR] 无法打开摄像头 /dev/video{self.camera_id}")
            sys.exit(1)
        print(f"[INFO] 摄像头已打开: {CAM_W}x{CAM_H}")

        threading.Thread(target=lidar_reader_loop,
                         args=(self.lidar_port, self.lidar_baud),
                         daemon=True).start()

        print(f"[INFO] 系统初始化完成")
        print(f"[INFO] 模型: {os.path.basename(self.model_path)}")
        print(f"[INFO] 舵机: {self.servo_port} @ {self.servo_baud}")
        print(f"[INFO] 激光雷达: {self.lidar_port} @ {self.lidar_baud}")
        print(f"[INFO] 安全距离: {WALL_MIN_DIST}m, 捕获阈值: {CAPTURE_BBOX_RATIO*100:.0f}%")

    def start_servo(self):
        try:
            self.servo = STM32ServoController(port=self.servo_port,
                                               baud=self.servo_baud)
            self.servo.send_mid()
            print(f"[INFO] 舵机已连接: {self.servo_port}")
        except Exception as e:
            print(f"[WARN] 舵机连接失败 ({e}), 仅推理模式")
            self.servo = None

    def send_motors(self, ch1, ch2, ch3, ch4):
        if self.servo:
            try:
                self.servo.send(int(ch1), int(ch2), int(ch3), int(ch4), status=0)
            except Exception:
                pass

    def run(self):
        self.setup()
        self.start_servo()
        time.sleep(0.5)

        self.state = STATE_SEARCH

        print("\n" + "=" * 50)
        print("[RUNNING] 总指挥启动, 按 Ctrl+C 停止")
        print("=" * 50 + "\n")

        bottle_bbox = None
        near_streak = 0

        while self.running:
            t_loop = time.time()

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_pad = letterbox(frame, INPUT_SIZE, PAD_COLOR)
            nv12_data = bgr2nv12(frame_pad)

            outputs = self.model.forward(nv12_data)
            detections = run_postprocess(self.lib, outputs, self.output_tensors,
                                         self.post_info)

            with lidar_lock:
                points = list(lidar_latest_points)

            is_obstacle, min_dist, near_pts = detect_obstacle(points)

            bottle_bbox = None
            bottle_conf = 0.0
            if detections:
                best = max(detections, key=lambda d: d['score'])
                if best['score'] >= CONF_THRES:
                    b = best['bbox']
                    bottle_bbox = (b[0], b[1], b[2], b[3])
                    bottle_conf = best['score']

            if is_obstacle and self.state not in (STATE_AVOID, STATE_CAPTURE):
                prev_state = self.state
                self.state = STATE_AVOID
                self.state_start = t_loop
                print(f"  [AVOID] 检测到障碍物! dist={min_dist:.2f}m, pts={near_pts}")
                self.send_motors(PWM_MID, AVOID_CH2, AVOID_CH3, PWM_MID)

            elif self.state == STATE_AVOID:
                if t_loop - self.state_start >= AVOID_DURATION:
                    self.state = STATE_SEARCH
                    self.send_motors(PWM_MID, PWM_MID, PWM_MID, PWM_MID)
                    print(f"  [AVOID] 避障完成, 恢复搜索")

            elif self.state == STATE_CAPTURE:
                if t_loop - self.state_start >= CAPTURE_DURATION:
                    self.state = STATE_SEARCH
                    self.send_motors(PWM_MID, PWM_MID, PWM_MID, PWM_MID)
                    print(f"  [CAPTURE] 捕获完成")

            elif self.state == STATE_APPROACH or self.state == STATE_SEARCH:
                ch1, ch2, ch3, ch4, target_state = compute_motors(
                    *bottle_bbox if bottle_bbox else (None, 0, None, 0),
                    CAM_W, CAM_H)

                if target_state == STATE_CAPTURE and bottle_bbox:
                    near_streak += 1
                    if near_streak >= CAPTURE_NEAR_FRAMES:
                        self.state = STATE_CAPTURE
                        self.state_start = t_loop
                        self.send_motors(ch1, ch2, ch3, ch4)
                        print(f"  [CAPTURE] 瓶子接近, 执行旋转捕获!")
                        near_streak = 0
                    else:
                        self.send_motors(PWM_MID, PWM_MID, PWM_MID, PWM_MID)
                elif target_state == STATE_APPROACH and bottle_bbox:
                    near_streak = 0
                    if self.state != STATE_APPROACH:
                        print(f"  [APPROACH] 发现瓶子 conf={bottle_conf:.2f}")
                    self.state = STATE_APPROACH
                    self.send_motors(ch1, ch2, ch3, ch4)
                else:
                    near_streak = 0
                    if self.state != STATE_SEARCH:
                        print(f"  [SEARCH] 无目标, 等待探测")
                    self.state = STATE_SEARCH
                    self.send_motors(PWM_MID, PWM_MID, PWM_MID, PWM_MID)

            elapsed = time.time() - t_loop
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)

    def shutdown(self):
        print("\n[INFO] 正在关闭...")
        self.running = False
        if self.servo:
            try:
                self.servo.send_mid()
                self.servo.close()
            except Exception:
                pass
        if self.cap:
            self.cap.release()
        print("[INFO] 系统已关闭")


# ==================== 入口 ====================

def main():
    global CONF_THRES, WALL_MIN_DIST, CAPTURE_BBOX_RATIO

    parser = argparse.ArgumentParser(description="RDK X5 Main Controller - 总指挥")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--cam", type=int, default=CAMERA_ID)
    parser.add_argument("--servo-port", default=SERVO_PORT)
    parser.add_argument("--servo-baud", type=int, default=SERVO_BAUD)
    parser.add_argument("--lidar-port", default=LIDAR_PORT)
    parser.add_argument("--lidar-baud", type=int, default=LIDAR_BAUD)
    parser.add_argument("--conf", type=float, default=CONF_THRES)
    parser.add_argument("--safe-dist", type=float, default=WALL_MIN_DIST)
    parser.add_argument("--capture-ratio", type=float, default=CAPTURE_BBOX_RATIO)
    args = parser.parse_args()

    CONF_THRES = args.conf
    WALL_MIN_DIST = args.safe_dist
    CAPTURE_BBOX_RATIO = args.capture_ratio

    controller = MainController(
        servo_port=args.servo_port,
        servo_baud=args.servo_baud,
        lidar_port=args.lidar_port,
        lidar_baud=args.lidar_baud,
        camera_id=args.cam,
        model_path=args.model,
    )

    def signal_handler(sig, frame):
        controller.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        controller.run()
    except KeyboardInterrupt:
        controller.shutdown()
    except Exception as e:
        print(f"[ERROR] {e}")
        controller.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
