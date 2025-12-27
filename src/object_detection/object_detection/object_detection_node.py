import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from cv_bridge import CvBridge
import cv2
import os

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.yolov8_model = YOLO("/home/meewoan/Comvi_ws/src/object_detection/object_detection/yolov8n.pt")
        self.class_names = self.yolov8_model.names
        self.class_names = list(self.class_names.values())
        print(self.class_names)
        self.bridge = CvBridge()
        self.subscriber = self.create_subscription(
            Image,
            'camera/image_raw',  # Replace with your image topic name
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Image, 'result/image_raw', 10)
        print("ObjectDetectionNode Process ID: ", os.getpid())


    def image_callback(self, msg):
        # self.get_logger().info('Received an image message')
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            visual_image = cv_image.copy()
            # Perform object detection
            results = self.yolov8_model.predict(cv_image, conf=0.5, classes=[0], show=False, verbose=False)
            for result in results:
                for box in result.boxes.cpu().numpy():
                    bbox = box.xyxy[0].astype(int)
                    x0, y0, x1, y1 = bbox
                    score = box.conf[0].astype(float)
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]
                    color = (0, 255, 0)  # Green color for bounding box
                    cv2.rectangle(visual_image, (x0, y0), (x1, y1), color, 2)

                    # # Dynamically adjust font scale and thickness based on bbox height
                    # bbox_height = y1 - y0
                    # font_scale = max(0.5, bbox_height / 100.0) 
                    # thickness = max(1, int(bbox_height / 50.0))
                    # cv2.putText(visual_image, class_name, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            result_msg = self.bridge.cv2_to_imgmsg(visual_image, encoding='bgr8')
            self.publisher.publish(result_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()