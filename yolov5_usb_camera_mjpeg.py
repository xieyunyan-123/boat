#!/usr/bin/env python3
"""
RDK X5 - YOLOv5 USB Camera + MJPEG 实时检测 (C++ 后处理优化版)
============================================================
使用 libpostprocess.so 替代 Python/NumPy 后处理，显著提升帧率。

用法:
    python3 yolov5_usb_camera_mjpeg.py [--model MODEL_PATH] [--cam N]
    python3 yolov5_usb_camera_mjpeg.py --model yolov5s_672x672_nv12.bin
    python3 yolov5_usb_camera_mjpeg.py --model best_detect_640x640_bayese_nv12.bin --num-classes 1

浏览器访问: http://<板子IP>:5000
"""

import numpy as np
import cv2
import ctypes
import json
import time
import threading
import sys
import os
import argparse
from flask import Flask, Response, render_template_string, jsonify

# ==================== 配置参数 ====================
MODEL_PATH   = "best_detect_640x640_bayese_nv12.bin"
INPUT_SIZE   = 640              # 模型输入边长 (自动从模型读取)
NUM_CLASSES  = 1                # 类别数 (自动从模型推导)
CONF_THRES   = 0.4              # 置信度阈值
NMS_THRES    = 0.45             # NMS IoU 阈值
NMS_TOP_K    = 500              # NMS 前 top-k 个框
IS_PAD_RESIZE = 1               # 1=letterbox, C++ 库自动处理坐标逆映射

CAMERA_ID    = 0                # USB 摄像头设备号
CAM_W        = 640              # 摄像头采集宽度
CAM_H        = 480              # 摄像头采集高度
CAM_FORMAT   = "YUYV"           # 摄像头格式: MJPG(压缩需CPU解) / YUYV(原始)
JPEG_QUALITY = 50               # MJPEG 推流质量 (10-100, 越低越快)
FPS_TARGET   = 30               # 推流目标帧率
STREAM_PORT  = 5000             # HTTP 服务端口

PAD_COLOR    = 114              # letterbox 灰色填充

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]


# ==================== C 结构体定义 ====================

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


# ==================== 全局状态 ====================
frame_lock = threading.Lock()
latest_frame = None       # 最新的带检测框的 BGR 图像 (用于 MJPEG 推流)
fps_info = {
    "cap_read_ms": 0.0,       # 摄像头读取耗时 ms
    "preprocess_ms": 0.0,    # 预处理耗时 ms (letterbox + NV12)
    "inference_ms": 0.0,     # BPU 推理耗时 ms
    "postprocess_ms": 0.0,   # C++ 后处理耗时 ms
    "draw_ms": 0.0,          # 绘制耗时 ms
    "overall_fps": 0.0,      # 总 FPS
    "detections": 0,         # 检出目标数
}


# ==================== 预处理 ====================

def letterbox(img_bgr, target=INPUT_SIZE, color=PAD_COLOR):
    """自适应 letterbox: 保持宽高比缩放到 target×target, 灰色填充。"""
    h0, w0 = img_bgr.shape[:2]
    r = target / max(h0, w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    if new_w != w0 or new_h != h0:
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = target - new_w
    dh = target - new_h
    left, top = dw // 2, dh // 2
    img_bgr = cv2.copyMakeBorder(
        img_bgr, top, dh - top, left, dw - left,
        cv2.BORDER_CONSTANT, value=(color, color, color))
    return img_bgr


def bgr2nv12(img_bgr):
    """BGR → NV12 一维字节数组。"""
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


# ==================== C++ 后处理接口 ====================

def get_tensor_layout(layout_str: str) -> int:
    if layout_str == "NCHW":
        return 2
    return 0  # NHWC or default


def init_postprocess_lib():
    """加载 libpostprocess.so 并设置函数签名。"""
    lib = ctypes.CDLL('/usr/lib/libpostprocess.so')

    # Yolov5doProcess: 逐层处理 (不设 argtypes, 与官方示例一致)
    # Yolov5PostProcess: 获取最终 NMS 结果 (JSON 字符串)
    lib.Yolov5PostProcess.argtypes = [ctypes.POINTER(Yolov5PostProcessInfo_t)]
    lib.Yolov5PostProcess.restype = ctypes.c_char_p

    return lib


def setup_output_tensors(model, lib, post_info):
    """初始化输出 tensor 结构体 (静态属性, 只需调用一次)。"""
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
    """
    对单帧推理结果执行 C++ 后处理 (解码 + NMS)。
    返回: list[dict], 每个 dict: {bbox, score, id, name}
    """
    out_count = len(outputs)

    # 更新每层输出 buffer 地址
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

        lib.Yolov5doProcess(
            output_tensors[i],
            ctypes.pointer(post_info),
            i)

    # 获取最终 NMS 结果
    result_bytes = lib.Yolov5PostProcess(ctypes.pointer(post_info))
    if not result_bytes:
        return []

    result_str = result_bytes.decode('utf-8', errors='replace')

    # JSON 字符串有固定前缀, 跳过前缀取 JSON 部分
    # 格式: "yolov5_postprocess: [{...}, ...]"
    idx = result_str.find('[')
    if idx < 0:
        return []

    try:
        data = json.loads(result_str[idx:])
    except json.JSONDecodeError:
        return []

    return data


# ==================== 绘制 ====================

def draw_detections(img_bgr, detections, class_names=None):
    """在 BGR 图像上绘制检测框和标签。"""
    for det in detections:
        bbox = det['bbox']       # [x1, y1, x2, y2]
        score = det['score']
        cls_id = int(det['id'])
        name = det.get('name', '')

        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 优先用自定义类名
        if class_names and cls_id < len(class_names):
            label = f"{class_names[cls_id]} {score:.2f}"
        elif name:
            label = f"{name} {score:.2f}"
        else:
            label = f"cls{cls_id} {score:.2f}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_bgr, (x1, y1 - th - baseline - 2),
                      (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img_bgr, label, (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img_bgr


# ==================== MJPEG 推流 ====================

app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RDK X5 YOLOv5 实时检测</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #111; color: #eee; font-family: Arial, sans-serif;
       display: flex; flex-direction: column; align-items: center;
       justify-content: center; min-height: 100vh; }
h1 { margin: 20px 0; font-size: 1.5rem; color: #0f0; }
.stream { border: 3px solid #0f0; border-radius: 8px; max-width: 95vw; }
.info { margin-top: 10px; font-size: 0.9rem; color: #888; }
.stats { margin-top: 5px; font-size: 0.75rem; color: #666; }
</style>
</head>
<body>
<h1>RDK X5 YOLOv5 · C++ 后处理</h1>
<img class="stream" src="/video_feed" alt="MJPEG Stream">
<div class="info">模型: {{model}} | 分辨率: {{w}}x{{h}} | C++ 后处理</div>
<div class="stats" id="stats">等待数据...</div>
<script>
setInterval(async () => {
  try {
    const r = await fetch('/stats');
    const s = await r.json();
    document.getElementById('stats').textContent =
      `FPS: ${s.overall_fps.toFixed(1)} | Pre: ${s.preprocess_ms.toFixed(0)}ms ` +
      `| Inf: ${s.inference_ms.toFixed(0)}ms | Post: ${s.postprocess_ms.toFixed(0)}ms ` +
      `| Draw: ${s.draw_ms.toFixed(0)}ms | Det: ${s.detections}`;
  } catch(e) {}
}, 2000);
</script>
</body>
</html>
""", model=os.path.basename(MODEL_PATH), w=CAM_W, h=CAM_H)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                display = latest_frame
            if display is None:
                time.sleep(0.05)
                continue

            # 叠加 FPS 信息
            cv2.putText(display,
                        f"FPS:{fps_info['overall_fps']:.1f} "
                        f"R:{fps_info['cap_read_ms']:.0f}ms "
                        f"I:{fps_info['inference_ms']:.0f}ms "
                        f"P:{fps_info['postprocess_ms']:.0f}ms "
                        f"D:{fps_info['detections']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 2)

            ret, jpeg = cv2.imencode(".jpg", display,
                                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ret:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpeg.tobytes() + b"\r\n")

            time.sleep(1.0 / FPS_TARGET)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    """返回 JSON 格式的实时性能统计。"""
    return jsonify(fps_info)


@app.route("/debug")
def debug():
    """调试: 检查 latest_frame 状态。"""
    with frame_lock:
        has_frame = latest_frame is not None
        shape = str(latest_frame.shape) if has_frame else "None"
    return jsonify({"latest_frame": shape, "has_frame": has_frame})


# ==================== 采集+推理线程 ====================

def capture_infer_loop(model, output_tensors, post_info, lib, class_names,
                       cam_id, cam_w, cam_h, cam_format):
    """
    主循环: USB 采集 → 预处理 → BPU 推理 → C++ 后处理 → 绘制。
    """
    global latest_frame, fps_info

    # ---- 打开 USB 摄像头 ----
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam_format))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # 注意: 不要设置 CAP_PROP_BUFFERSIZE=1, 在此平台上反而导致严重性能下降

    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 /dev/video{cam_id}", flush=True)
        return

    print(f"[INFO] 摄像头已打开: {cam_w}×{cam_h} @ {cam_format}", flush=True)
    print(f"[INFO] 模型: {os.path.basename(MODEL_PATH)}", flush=True)
    print(f"[INFO] 类别数: {NUM_CLASSES}, 输入尺寸: {INPUT_SIZE}×{INPUT_SIZE}", flush=True)
    print(f"[INFO] 后处理: libpostprocess.so (C++)", flush=True)
    print(f"\n[READY] 浏览器打开: http://<板子IP>:{STREAM_PORT}\n", flush=True)

    # ---- 帧率统计 ----
    t_overall_start = time.time()
    overall_count = 0
    cap_read_sum = 0.0
    preprocess_sum = 0.0
    inference_sum = 0.0
    postprocess_sum = 0.0
    draw_sum = 0.0
    stat_count = 0

    while True:
        # 1) 采集
        t0 = time.time()
        ret, frame = cap.read()
        t_cap_done = time.time()

        if not ret:
            time.sleep(0.01)
            continue

        # 2) 预处理: letterbox + BGR→NV12
        frame_pad = letterbox(frame, INPUT_SIZE, PAD_COLOR)
        nv12_data = bgr2nv12(frame_pad)
        t_preprocess = time.time()

        # 3) BPU 推理
        outputs = model.forward(nv12_data)
        t_inference = time.time()

        # 4) C++ 后处理 (解码 + NMS)
        detections = run_postprocess(lib, outputs, output_tensors, post_info)
        t_postprocess = time.time()

        # 5) 绘制检测框 (在原图上直接画)
        draw_detections(frame, detections, class_names)
        t_draw = time.time()

        # ---- 累加耗时 ----
        cap_read_sum    += (t_cap_done - t0) * 1000
        preprocess_sum  += (t_preprocess - t_cap_done) * 1000
        inference_sum   += (t_inference - t_preprocess) * 1000
        postprocess_sum += (t_postprocess - t_inference) * 1000
        draw_sum        += (t_draw - t_postprocess) * 1000
        stat_count      += 1

        overall_count += 1
        if time.time() - t_overall_start >= 1.0:
            fps_info["cap_read_ms"] = cap_read_sum / stat_count
            fps_info["preprocess_ms"] = preprocess_sum / stat_count
            fps_info["inference_ms"] = inference_sum / stat_count
            fps_info["postprocess_ms"] = postprocess_sum / stat_count
            fps_info["draw_ms"] = draw_sum / stat_count
            fps_info["overall_fps"] = overall_count / (time.time() - t_overall_start)
            fps_info["detections"] = len(detections)
            cap_read_sum = 0.0
            preprocess_sum = 0.0
            inference_sum = 0.0
            postprocess_sum = 0.0
            draw_sum = 0.0
            stat_count = 0
            overall_count = 0
            t_overall_start = time.time()

        # 6) 更新全局帧 (供 MJPEG 推流线程读取)
        with frame_lock:
            latest_frame = frame.copy()


# ==================== 入口 ====================

def main():
    global MODEL_PATH, INPUT_SIZE, NUM_CLASSES, IS_PAD_RESIZE
    global CAMERA_ID, CAM_W, CAM_H, CAM_FORMAT
    global JPEG_QUALITY, STREAM_PORT

    parser = argparse.ArgumentParser(description="RDK X5 YOLOv5 USB Camera MJPEG")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"模型文件路径 (默认: {MODEL_PATH})")
    parser.add_argument("--cam", type=int, default=CAMERA_ID,
                        help=f"USB 摄像头设备号 (默认: {CAMERA_ID})")
    parser.add_argument("--cam-w", type=int, default=None,
                        help=f"摄像头宽度 (默认: {CAM_W})")
    parser.add_argument("--cam-h", type=int, default=None,
                        help=f"摄像头高度 (默认: {CAM_H})")
    parser.add_argument("--conf", type=float, default=CONF_THRES,
                        help=f"置信度阈值 (默认: {CONF_THRES})")
    parser.add_argument("--nms", type=float, default=NMS_THRES,
                        help=f"NMS 阈值 (默认: {NMS_THRES})")
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY,
                        help=f"JPEG 质量 1-100 (默认: {JPEG_QUALITY})")
    parser.add_argument("--cam-format", default=CAM_FORMAT,
                        help=f"摄像头格式: MJPG / YUYV (默认: {CAM_FORMAT})")
    parser.add_argument("--port", type=int, default=STREAM_PORT,
                        help=f"HTTP 服务端口 (默认: {STREAM_PORT})")
    parser.add_argument("--no-pad", action="store_true",
                        help="直接拉伸缩放而非 letterbox (is_pad_resize=0)")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="类别数 (默认从模型推导)")
    parser.add_argument("--class-names", nargs="*", default=None,
                        help="自定义类别名")
    args = parser.parse_args()

    MODEL_PATH = args.model
    if args.cam_w is not None: CAM_W = args.cam_w
    if args.cam_h is not None: CAM_H = args.cam_h
    CAMERA_ID = args.cam
    CAM_FORMAT = args.cam_format
    CONF_THRES_local = args.conf
    NMS_THRES_local = args.nms
    JPEG_QUALITY = args.jpeg_quality
    STREAM_PORT = args.port
    IS_PAD_RESIZE = 0 if args.no_pad else 1

    # ---- 加载模型 ----
    print(f"[INFO] 加载模型: {MODEL_PATH}")
    try:
        from hobot_dnn import pyeasy_dnn as dnn
        models = dnn.load(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        sys.exit(1)
    model = models[0]

    # ---- 推导模型参数 ----
    inp = model.inputs[0]
    if inp.properties.layout == "NCHW":
        INPUT_SIZE = inp.properties.shape[2]
    else:
        INPUT_SIZE = inp.properties.shape[1]
    print(f"  输入: layout={inp.properties.layout}, shape={inp.properties.shape}, "
          f"推断输入边长={INPUT_SIZE}")

    NA = 3  # YOLOv5 固定 3 个 anchor
    if args.num_classes is not None:
        NUM_CLASSES = args.num_classes
    else:
        NUM_CLASSES = model.outputs[0].properties.shape[-1] // NA - 5
    print(f"  输出层数: {len(model.outputs)}, 推导类别数: {NUM_CLASSES}")

    for i, out in enumerate(model.outputs):
        print(f"  输出[{i}]: shape={out.properties.shape}, "
              f"scale_len={len(out.properties.scale_data)}")

    # ---- 设置类名 ----
    class_names = args.class_names if args.class_names else COCO_NAMES[:NUM_CLASSES]

    # ---- 初始化 C++ 后处理 ----
    lib = init_postprocess_lib()

    post_info = Yolov5PostProcessInfo_t()
    post_info.height = INPUT_SIZE
    post_info.width = INPUT_SIZE
    post_info.ori_height = CAM_H
    post_info.ori_width = CAM_W
    post_info.score_threshold = CONF_THRES_local
    post_info.nms_threshold = NMS_THRES_local
    post_info.nms_top_k = NMS_TOP_K
    post_info.is_pad_resize = IS_PAD_RESIZE

    output_tensors = setup_output_tensors(model, lib, post_info)

    # ---- 启动采集+推理线程 ----
    infer_thread = threading.Thread(
        target=capture_infer_loop,
        args=(model, output_tensors, post_info, lib, class_names,
              CAMERA_ID, CAM_W, CAM_H, CAM_FORMAT),
        daemon=True)
    infer_thread.start()

    # ---- 启动 Flask MJPEG 服务 ----
    print(f"\n[READY] 浏览器打开: http://<板子IP>:{STREAM_PORT}\n")
    app.run(host="0.0.0.0", port=STREAM_PORT, debug=False, threaded=True)
    print("[INFO] Flask 已退出", flush=True)


if __name__ == "__main__":
    main()
