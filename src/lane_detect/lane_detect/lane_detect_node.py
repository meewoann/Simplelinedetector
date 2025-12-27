import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import time
import configparser

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge

from lane_detect.preprocess import *

class LaneDetectorNode(Node):

    def __init__(self):
        super().__init__('lane_detector')

        self.pub_debug_img = self.create_publisher(
            Image,
            '/lane/debug_image',
            10
        )

        # ===== Load config =====
        config_path = self.declare_parameter(
            'config_path',
            '/home/meewoan/Comvi_ws/src/lane_detect/config.ini'
        ).value

        self.cfg = load_config(config_path)

        # ===== ROS =====
        self.bridge = CvBridge()

        self.sub_img = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.pub_junction = self.create_publisher(
            Int32,
            '/lane/junction_state',
            10
        )

        # ===== lane offset publisher =====
        self.pub_lane_offset = self.create_publisher(
            Int32,
            '/lane/offset',
            10
        )

        # ===== State =====
        self.lane_memory = (None, None, None)
        self.timer_state = (False, 0.0)

        self.get_logger().info("Lane Detector Node Started")

    # ================= CALLBACK =================

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = cv2.resize(
            frame,
            (self.cfg["FRAME_W"], self.cfg["FRAME_H"])
        )

        thresh = preprocess_frame(frame)

        detected, left, right, center = extract_lane_edges(
            thresh,
            self.cfg["SCAN_ROW"],
            self.lane_memory
        )

        if not detected and self.lane_memory[0] is None:
            return

        self.lane_memory = (left, right, center)

        is_junction = detect_junction(left, right, self.cfg)
        raw_state = 1 if is_junction else 0
        state, self.timer_state = junction_hold_logic(
                                    raw_state,
                                    self.timer_state,
                                    self.cfg)

        # ===== Publish junction =====
        msg_state = Int32()
        msg_state.data = state
        self.pub_junction.publish(msg_state)

        # ===== NEW: compare lane center with frame center =====
        frame_center_x = self.cfg["FRAME_W"] // 2
        lane_offset = center - frame_center_x

        if abs(lane_offset) <= self.cfg["CENTER_THRESHOLD"]:
            offset_state = 0      # CENTER
        elif lane_offset < 0:
            offset_state = -1     # LEFT
        else:
            offset_state = 1      # RIGHT

        msg_offset = Int32()
        msg_offset.data = offset_state
        self.pub_lane_offset.publish(msg_offset)

        # ===== Debug visualization =====
        cv2.line(
            frame,
            (0, self.cfg["SCAN_ROW"]),
            (self.cfg["FRAME_W"], self.cfg["SCAN_ROW"]),
            (255, 255, 0), 1
        )

        # Lane edges & center
        cv2.circle(frame, (left, self.cfg["SCAN_ROW"]), 5, (0, 255, 0), -1)
        cv2.circle(frame, (right, self.cfg["SCAN_ROW"]), 5, (0, 255, 0), -1)
        cv2.circle(frame, (center, self.cfg["SCAN_ROW"]), 8, (255, 0, 0), -1)

        # Frame center
        cv2.circle(
            frame,
            (frame_center_x, self.cfg["SCAN_ROW"]),
            8,
            (0, 0, 255),
            -1
        )

        # Offset vector
        cv2.line(
            frame,
            (frame_center_x, self.cfg["SCAN_ROW"]),
            (center, self.cfg["SCAN_ROW"]),
            (0, 255, 255),
            2
        )

        # Text
        label = "CENTER" if offset_state == 0 else ("LEFT" if offset_state < 0 else "RIGHT")
        cv2.putText(
            frame,
            f"Lane Offset: {label} ({lane_offset})",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if offset_state == 0 else (0, 0, 255),
            2
        )

        cv2.putText(frame, f"Junction: {state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub_debug_img.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
