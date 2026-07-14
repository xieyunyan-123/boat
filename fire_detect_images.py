#!/usr/bin/env python3
"""使用 fire_detect.bin 模型对 image/ 中的图片进行火灾检测 (YOLOv8 DFL 后处理)"""

import numpy as np
import cv2
import os
import sys
import glob

MODEL_PATH  = "fire_detect.bin"
IMAGE_DIR   = "image"
RESULT_DIR  = "result"
CONF_THRES  = 0.25
NMS_THRES   = 0.45
MAX_DET     = 300
REG_MAX     = 16
STRIDES     = [8, 16, 32]
PAD_COLOR   = 114


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
    return img_bgr, (left, top, r)


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


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def make_grid_points(h, w, stride):
    yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    return np.stack([xv, yv], axis=-1).astype(np.float32) * stride


def dfl_decode(bbox_reg, reg_max=REG_MAX):
    b, c, h, w = bbox_reg.shape
    reg = bbox_reg.reshape(b, 4, reg_max, h, w)
    reg = reg.transpose(0, 1, 3, 4, 2)  # (b, 4, h, w, reg_max)
    prob = softmax(reg, axis=-1)
    indices = np.arange(reg_max, dtype=np.float32).reshape(1, 1, 1, 1, -1)
    dist = (prob * indices).sum(axis=-1)  # (b, 4, h, w)
    return dist


def dist2bbox(dist, anchor_points, stride):
    lt, rb = dist[:, :2], dist[:, 2:]
    x1 = anchor_points[..., 0:1] - lt[:, 0:1]
    y1 = anchor_points[..., 1:2] - lt[:, 1:2]
    x2 = anchor_points[..., 0:1] + rb[:, 0:1]
    y2 = anchor_points[..., 1:2] + rb[:, 1:2]
    return np.concatenate([x1, y1, x2, y2], axis=1)


def nms(boxes, scores, thresh):
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)


def postprocess(outputs, input_size, ori_shape, conf_thres=CONF_THRES, nms_thres=NMS_THRES):
    ori_h, ori_w = ori_shape[:2]

    all_boxes = []
    all_scores = []

    for idx, stride in enumerate(STRIDES):
        bbox_out = outputs[idx * 2].buffer      # (1, 64, h, w) NCHW
        cls_out = outputs[idx * 2 + 1].buffer   # (1, 1, h, w) NCHW

        _, _, h, w = bbox_out.shape

        dist = dfl_decode(bbox_out, REG_MAX)     # (1, 4, h, w)
        dist = dist.reshape(1, 4, h * w).transpose(0, 2, 1).reshape(-1, 4)  # (hw, 4)

        ap = make_grid_points(h, w, stride).reshape(-1, 2)  # (hw, 2)

        # ltrb -> xyxy
        lt, rb = dist[:, :2], dist[:, 2:]
        x1 = ap[:, 0] - lt[:, 0] * stride
        y1 = ap[:, 1] - lt[:, 1] * stride
        x2 = ap[:, 0] + rb[:, 0] * stride
        y2 = ap[:, 1] + rb[:, 1] * stride

        boxes = np.stack([x1, y1, x2, y2], axis=1)  # (hw, 4)

        scores = cls_out.reshape(-1)  # (hw,)
        scores = 1.0 / (1.0 + np.exp(-scores))  # sigmoid

        all_boxes.append(boxes)
        all_scores.append(scores)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    # Filter by confidence
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return []

    # Map from 640x640 letterbox space to original image space
    # letterbox: scale then pad
    r = input_size / max(ori_w, ori_h)
    pad_x = (input_size - ori_w * r) / 2
    pad_y = (input_size - ori_h * r) / 2

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / r

    # Clip to image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ori_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, ori_h)

    # NMS
    keep = nms(boxes, scores, nms_thres)
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) > MAX_DET:
        idx = np.argsort(scores)[::-1][:MAX_DET]
        boxes = boxes[idx]
        scores = scores[idx]

    detections = []
    for i in range(len(boxes)):
        detections.append({
            'bbox': boxes[i].tolist(),
            'score': float(scores[i]),
            'id': 0,
            'name': 'fire',
        })
    return detections


def draw_detections(img_bgr, detections):
    for det in detections:
        bbox = det['bbox']
        score = det['score']
        x1, y1, x2, y2 = [int(v) for v in bbox]

        color = (0, 0, 255)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        label = f"fire {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, y1 - th - baseline - 2),
                      (x1 + tw, y1), color, -1)
        cv2.putText(img_bgr, label, (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img_bgr


def main():
    workdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workdir)

    model_full = os.path.join(workdir, MODEL_PATH)
    image_full = os.path.join(workdir, IMAGE_DIR)
    result_full = os.path.join(workdir, RESULT_DIR)

    if not os.path.exists(model_full):
        print(f"[ERROR] 模型文件不存在: {model_full}")
        sys.exit(1)

    os.makedirs(result_full, exist_ok=True)

    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        for f in glob.glob(os.path.join(image_full, ext)):
            image_files.append(f)
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"[ERROR] 在 {image_full} 中未找到图片文件")
        sys.exit(1)

    print(f"[INFO] 加载模型: {model_full}")
    from hobot_dnn import pyeasy_dnn as dnn
    models = dnn.load(model_full)
    model = models[0]

    inp = model.inputs[0]
    if inp.properties.layout == "NCHW":
        input_size = inp.properties.shape[2]
    else:
        input_size = inp.properties.shape[1]
    print(f"  输入边长: {input_size}, layout: {inp.properties.layout}")

    num_outputs = len(model.outputs)
    print(f"  输出层数: {num_outputs}")

    for i, out in enumerate(model.outputs):
        sh = out.properties.shape
        print(f"    output[{i}]: shape={sh}, scale_len={len(out.properties.scale_data)}")

    print(f"\n[INFO] 开始检测 {len(image_files)} 张图片...\n")

    for img_path in image_files:
        basename = os.path.basename(img_path)
        print(f"  处理: {basename}")

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [SKIP] 无法读取: {img_path}")
            continue

        ori_h, ori_w = frame.shape[:2]

        frame_pad, (pad_left, pad_top, ratio) = letterbox(frame, input_size, PAD_COLOR)
        nv12_data = bgr2nv12(frame_pad)

        outputs = model.forward(nv12_data)
        detections = postprocess(outputs, input_size, (ori_h, ori_w),
                                 conf_thres=CONF_THRES, nms_thres=NMS_THRES)

        if detections:
            print(f"    检出 {len(detections)} 个目标:")
            for d in detections:
                bbox = d['bbox']
                print(f"      {d['name']} score={d['score']:.3f} "
                      f"bbox={[int(v) for v in bbox]}")
        else:
            print(f"    未检出目标")

        draw_detections(frame, detections)

        result_name = f"detected_{basename}"
        result_path = os.path.join(result_full, result_name)
        cv2.imwrite(result_path, frame)
        print(f"    结果保存: {result_path}\n")

    print(f"[DONE] 全部完成, 结果保存在 {result_full}/")


if __name__ == "__main__":
    main()
