#!/usr/bin/env python3
"""
RDK X5 - 火焰检测 USB Camera + MJPEG 实时推流 (C 后处理加速版)
==============================================================
使用 fire_detect.bin + libfire_postprocess.so (C 实现 YOLOv8 DFL + NMS)

用法:
    python3 fire_mjpeg_stream.py [--cam N] [--port 5000]
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

MODEL_PATH    = "fire_detect.bin"
LIB_PATH      = "./libfire_postprocess.so"
INPUT_SIZE    = 640
CONF_THRES    = 0.25
NMS_THRES     = 0.45

CAMERA_ID     = 0
CAM_W         = 640
CAM_H         = 480
CAM_FORMAT    = "YUYV"
JPEG_QUALITY  = 50
FPS_TARGET    = 30
STREAM_PORT   = 5000
PAD_COLOR     = 114

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


# ==================== 绘制 ====================

def draw_detections(img_bgr, detections):
    for det in detections:
        bbox = det['bbox']
        score = det['score']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"fire {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, y1 - th - baseline - 2),
                      (x1 + tw, y1), (0, 0, 255), -1)
        cv2.putText(img_bgr, label, (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img_bgr


# ==================== Flask MJPEG ====================

app = Flask(__name__)

frame_lock = threading.Lock()
latest_frame = None
fps_info = {
    "overall_fps": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0,
    "preprocess_ms": 0.0, "detections": 0,
}


@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fire Detection · C PostProcess</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #111; color: #eee; font-family: Arial, sans-serif;
       display: flex; flex-direction: column; align-items: center;
       justify-content: center; min-height: 100vh; }
h1 { margin: 20px 0; font-size: 1.5rem; color: #f44; }
.stream { border: 3px solid #f44; border-radius: 8px; max-width: 95vw; }
.stats { margin-top: 5px; font-size: 0.8rem; color: #aaa; }
</style>
</head>
<body>
<h1>RDK X5 Fire Detection (C PostProcess)</h1>
<img class="stream" src="/video_feed" alt="MJPEG Stream">
<div class="stats" id="stats">waiting...</div>
<script>
setInterval(async () => {
  try {
    const r = await fetch('/stats');
    const s = await r.json();
    document.getElementById('stats').textContent =
      `FPS: ${s.overall_fps.toFixed(1)} | Pre: ${s.preprocess_ms.toFixed(0)}ms ` +
      `| Inf: ${s.inference_ms.toFixed(0)}ms | Post(C): ${s.postprocess_ms.toFixed(0)}ms ` +
      `| Det: ${s.detections}`;
  } catch(e) {}
}, 1000);
</script>
</body>
</html>
""", w=CAM_W, h=CAM_H)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                display = latest_frame
            if display is None:
                time.sleep(0.05)
                continue
            cv2.putText(display,
                        f"FPS:{fps_info['overall_fps']:.1f} "
                        f"I:{fps_info['inference_ms']:.0f}ms "
                        f"C:{fps_info['postprocess_ms']:.0f}ms "
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
    return jsonify(fps_info)


# ==================== 采集 + 推理 ====================

def capture_infer_loop(model, cam_id, cam_w, cam_h, cam_format, input_size,
                       conf_thres, nms_thres):
    global latest_frame, fps_info

    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam_format))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 /dev/video{cam_id}", flush=True)
        return

    print(f"[INFO] 摄像头: {cam_w}x{cam_h} @ {cam_format}", flush=True)
    print(f"[INFO] 模型: fire_detect.bin + C postprocess", flush=True)
    print(f"[INFO] 输入: {input_size}x{input_size}, conf={conf_thres}", flush=True)
    print(f"\n[READY] http://<IP>:{STREAM_PORT}\n", flush=True)

    t_start = time.time()
    frame_count = 0
    inf_sum = 0.0
    pre_sum = 0.0
    post_sum = 0.0
    stat_count = 0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_pad = letterbox(frame, input_size)
        nv12_data = bgr2nv12(frame_pad)
        t_pre = time.time()

        outputs = model.forward(nv12_data)
        t_inf = time.time()

        detections = c_postprocess(outputs, input_size, cam_w, cam_h,
                                   conf_thres, nms_thres)
        t_post = time.time()

        draw_detections(frame, detections)

        pre_sum += (t_pre - t0) * 1000
        inf_sum += (t_inf - t_pre) * 1000
        post_sum += (t_post - t_inf) * 1000
        stat_count += 1

        frame_count += 1
        dt = time.time() - t_start
        if dt >= 1.0:
            fps_info["overall_fps"] = frame_count / dt
            fps_info["preprocess_ms"] = pre_sum / stat_count
            fps_info["inference_ms"] = inf_sum / stat_count
            fps_info["postprocess_ms"] = post_sum / stat_count
            fps_info["detections"] = len(detections)
            t_start = time.time()
            frame_count = 0
            pre_sum = 0.0
            inf_sum = 0.0
            post_sum = 0.0
            stat_count = 0

        with frame_lock:
            latest_frame = frame.copy()


# ==================== 入口 ====================

def main():
    global MODEL_PATH, INPUT_SIZE, CAMERA_ID, CAM_W, CAM_H, CAM_FORMAT
    global JPEG_QUALITY, STREAM_PORT, LIB_PATH, _pp

    parser = argparse.ArgumentParser(description="Fire Detection MJPEG (C PostProcess)")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--cam", type=int, default=CAMERA_ID)
    parser.add_argument("--cam-w", type=int, default=None)
    parser.add_argument("--cam-h", type=int, default=None)
    parser.add_argument("--conf", type=float, default=CONF_THRES)
    parser.add_argument("--nms", type=float, default=NMS_THRES)
    parser.add_argument("--port", type=int, default=STREAM_PORT)
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--cam-format", default=CAM_FORMAT)
    args = parser.parse_args()

    MODEL_PATH = args.model
    if args.cam_w is not None: CAM_W = args.cam_w
    if args.cam_h is not None: CAM_H = args.cam_h
    CAMERA_ID = args.cam
    CAM_FORMAT = args.cam_format
    STREAM_PORT = args.port
    JPEG_QUALITY = args.jpeg_quality

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

    for i, out in enumerate(model.outputs):
        print(f"  输出[{i}]: {out.properties.shape}")

    infer_thread = threading.Thread(
        target=capture_infer_loop,
        args=(model, CAMERA_ID, CAM_W, CAM_H, CAM_FORMAT, INPUT_SIZE,
              args.conf, args.nms),
        daemon=True)
    infer_thread.start()

    app.run(host="0.0.0.0", port=STREAM_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
