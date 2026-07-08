#!/usr/bin/env python3
"""Camera + YOLOv5 (BPU) detection node.

Captures frames from a V4L2 camera, runs the YOLOv5 bottle detector on the
Horizon BPU and publishes the best detection of each frame on
``~/bottle_detection`` (rdk_interfaces/BottleDetection). Optionally publishes
the annotated frame on ``~/image_annotated`` (sensor_msgs/Image).
"""

import os

import cv2
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from rdk_interfaces.msg import BottleDetection

from rdk_bottle_hunter.yolov5_bpu import Yolov5Detector


class CameraDetectorNode(Node):

    def __init__(self):
        super().__init__('camera_detector_node')

        default_model = os.path.join(
            get_package_share_directory('rdk_bottle_hunter'),
            'model', 'best_detect_640x640_bayese_nv12.bin')

        self.declare_parameter('model_path', default_model)
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('publish_annotated', False)

        model_path = self.get_parameter('model_path').value
        self.camera_id = self.get_parameter('camera_id').value
        self.cam_w = self.get_parameter('frame_width').value
        self.cam_h = self.get_parameter('frame_height').value
        fps = self.get_parameter('fps').value
        self.conf_thres = self.get_parameter('conf_threshold').value
        nms_thres = self.get_parameter('nms_threshold').value
        self.publish_annotated = self.get_parameter('publish_annotated').value

        if not os.path.exists(model_path):
            self.get_logger().error(f'Model not found: {model_path}')
            raise FileNotFoundError(model_path)

        self.get_logger().info(f'Loading YOLOv5 model: {model_path}')
        self.detector = Yolov5Detector(model_path,
                                       conf_thres=self.conf_thres,
                                       nms_thres=nms_thres)
        self.get_logger().info(f'Model ready, input size {self.detector.input_size}')

        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera /dev/video{self.camera_id}')
            raise RuntimeError('camera open failed')
        self.get_logger().info(f'Camera opened: {self.cam_w}x{self.cam_h}')

        self.det_pub = self.create_publisher(BottleDetection, 'bottle_detection', 10)
        self.img_pub = None
        if self.publish_annotated:
            self.img_pub = self.create_publisher(Image, 'image_annotated', 1)

        period = 1.0 / max(1, fps)
        self.timer = self.create_timer(period, self.on_timer)

    def on_timer(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        detections = self.detector.detect(frame)

        msg = BottleDetection()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.image_width = int(self.cam_w)
        msg.image_height = int(self.cam_h)

        best = None
        if detections:
            best = max(detections, key=lambda d: d['score'])
            if best['score'] < self.conf_thres:
                best = None

        if best is not None:
            b = best['bbox']
            msg.detected = True
            msg.score = float(best['score'])
            msg.x_min, msg.y_min, msg.x_max, msg.y_max = (
                float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        else:
            msg.detected = False

        self.det_pub.publish(msg)

        if self.img_pub is not None:
            if best is not None:
                b = best['bbox']
                cv2.rectangle(frame, (int(b[0]), int(b[1])),
                              (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"bottle {best['score']:.2f}",
                            (int(b[0]), int(b[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.img_pub.publish(self._to_image_msg(frame, msg.header))

    @staticmethod
    def _to_image_msg(frame_bgr, header):
        img = Image()
        img.header = header
        img.height, img.width = frame_bgr.shape[:2]
        img.encoding = 'bgr8'
        img.is_bigendian = 0
        img.step = int(frame_bgr.shape[1] * 3)
        img.data = frame_bgr.tobytes()
        return img

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
