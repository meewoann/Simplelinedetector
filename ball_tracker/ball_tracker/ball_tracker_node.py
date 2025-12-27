import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class BallTrackerNode(Node):
    def __init__(self):
        super().__init__('ball_tracker_node')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        self.image_pub = self.create_publisher(Image, '/ball/image_processed', 10)
        self.coord_pub = self.create_publisher(String, '/ball/ball_string', 10)

        self.cmd_pub = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)

        self.get_logger().info("✅ Ball Tracker using Contour started.")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Lọc màu xanh lá
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 60, 60])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Xử lý nhiễu
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        twist = Twist()
        ball_detected = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 200:
                (x, y), radius = cv2.minEnclosingCircle(largest)
                x, y, radius = int(x), int(y), int(radius)

                # Vẽ lên ảnh
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

                x_relative = x - center_x
                y_relative = center_y - y

                # Gửi thông tin vị trí bóng
                msg_str = String()
                msg_str.data = f"x: {x_relative}, y: {y_relative}, r: {radius}"
                self.coord_pub.publish(msg_str)
                self.get_logger().info(f"[Contour] {msg_str.data}")

                ball_detected = True

                # ==== Điều khiển ====
                if radius >= 40 and -50 <= x_relative <= 50:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.get_logger().info("✅ Ball close and centered → Stop")
                elif not (-50 <= x_relative <= 50):
                    twist.linear.x = 0.0
                    twist.angular.z = -0.5 if x_relative > 0 else 0.5
                    self.get_logger().info("🔄 Ball detected but not centered → Turning to center")
                else:
                    twist.linear.x = 0.5
                    twist.angular.z = 0.0
                    self.get_logger().info("🚀 Ball centered but far → Moving forward")

        if not ball_detected:
            twist.linear.x = 0.0
            twist.angular.z = 0.5  # quay nhẹ để tìm
            self.get_logger().info("❌ No ball detected → Rotating to search")

        # Gửi lệnh điều khiển
        self.cmd_pub.publish(twist)

        # Gửi ảnh kết quả
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image_msg.header = msg.header
        self.image_pub.publish(image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BallTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

