#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import time
import numpy as np
from cv_bridge import CvBridge
import os

class CameraReaderNode(Node):
    def __init__(self):
        super().__init__('camera_reader_node')

        self.publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.timer = self.create_timer(0.033, self.timer_callback)  # ~30 FPS
        self.bridge = CvBridge()

        self.video_path = r'/home/meewoan/Comvi_ws/vid/riel1.mp4'
        self.cap = cv2.VideoCapture(self.video_path)

        self.prev_time = time.time()
        print("CameraReaderNode Process ID:", os.getpid())

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open video file")
            rclpy.shutdown()

    def timer_callback(self):
        ret, frame = self.cap.read()

        # Nếu đọc fail (chạy hết video) → quay lại frame đầu
        if not ret:
            self.get_logger().info("Video ended, looping...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return  

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)

        # Debug FPS 
        # cur_time = time.time()
        # fps = 1.0 / (cur_time - self.prev_time)
        # self.get_logger().info(f"FPS: {fps:.2f}")
        # self.prev_time = cur_time

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraReaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
