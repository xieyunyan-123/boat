"""YOLOv5 helpers for the Horizon RDK X5 BPU (hobot_dnn + libpostprocess.so).

This module is a thin, node-agnostic wrapper around the ctypes-based
post-processing that was previously inlined in ``main.py``.
"""

import ctypes
import json

import cv2
import numpy as np

POSTPROCESS_LIB = '/usr/lib/libpostprocess.so'


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


def letterbox(img_bgr, target, color=114):
    """Resize keeping aspect ratio and pad to a square ``target`` size."""
    h0, w0 = img_bgr.shape[:2]
    r = target / max(h0, w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    if new_w != w0 or new_h != h0:
        img_bgr = cv2.resize(img_bgr, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
    dw, dh = target - new_w, target - new_h
    left, top = dw // 2, dh // 2
    img_bgr = cv2.copyMakeBorder(img_bgr, top, dh - top, left, dw - left,
                                 cv2.BORDER_CONSTANT, value=(color, color, color))
    return img_bgr


def bgr2nv12(img_bgr):
    """Convert a BGR image to NV12 (the BPU input layout)."""
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


def _tensor_layout(layout_str):
    return 2 if layout_str == "NCHW" else 0


class Yolov5Detector:
    """Loads a YOLOv5 ``.bin`` model on the BPU and runs inference + NMS."""

    def __init__(self, model_path, conf_thres=0.4, nms_thres=0.45,
                 nms_top_k=500, is_pad_resize=1, pad_color=114):
        from hobot_dnn import pyeasy_dnn as dnn

        self.pad_color = pad_color
        self._models = dnn.load(model_path)
        self.model = self._models[0]

        inp = self.model.inputs[0]
        if inp.properties.layout == "NCHW":
            self.input_size = inp.properties.shape[2]
        else:
            self.input_size = inp.properties.shape[1]

        self.lib = ctypes.CDLL(POSTPROCESS_LIB)
        self.lib.Yolov5PostProcess.argtypes = [ctypes.POINTER(Yolov5PostProcessInfo_t)]
        self.lib.Yolov5PostProcess.restype = ctypes.c_char_p

        self.output_tensors = self._setup_output_tensors()
        self.post_info = Yolov5PostProcessInfo_t()
        self.post_info.height = self.input_size
        self.post_info.width = self.input_size
        self.post_info.score_threshold = conf_thres
        self.post_info.nms_threshold = nms_thres
        self.post_info.nms_top_k = nms_top_k
        self.post_info.is_pad_resize = is_pad_resize

    def _setup_output_tensors(self):
        out_count = len(self.model.outputs)
        tensors = (hbDNNTensor_t * out_count)()
        for i in range(out_count):
            out_prop = self.model.outputs[i].properties
            tensors[i].properties.tensorLayout = _tensor_layout(out_prop.layout)
            if len(out_prop.scale_data) > 0:
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

    def detect(self, frame_bgr):
        """Run detection on a BGR frame. Returns a list of dicts with
        keys ``bbox`` ([x1,y1,x2,y2]) and ``score``."""
        ori_h, ori_w = frame_bgr.shape[:2]
        self.post_info.ori_height = ori_h
        self.post_info.ori_width = ori_w

        frame_pad = letterbox(frame_bgr, self.input_size, self.pad_color)
        nv12_data = bgr2nv12(frame_pad)
        outputs = self.model.forward(nv12_data)

        for i in range(len(outputs)):
            if self.output_tensors[i].properties.quantiType == 0:
                ptr = ctypes.cast(
                    outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_void_p)
            else:
                ptr = ctypes.cast(
                    outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    ctypes.c_void_p)
            self.output_tensors[i].sysMem[0].virAddr = ptr
            self.lib.Yolov5doProcess(self.output_tensors[i],
                                     ctypes.pointer(self.post_info), i)

        result_bytes = self.lib.Yolov5PostProcess(ctypes.pointer(self.post_info))
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
