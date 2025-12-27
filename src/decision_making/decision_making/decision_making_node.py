import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ActionDecisionNode(Node):
    def __init__(self):
        super().__init__('action_decision_node')
        self.subscriber = self.create_subscription(
            String,
            'detected_objects',
            self.listener_callback,
            10)
        self.get_logger().info("listening on 'detected_objects'")

    def listener_callback(self, msg: String):
        self.get_logger().info(f"Received message: {msg.data}")

        data = msg.data
        if not data.startswith("Detected: "):
            return
        data = data[len("Detected: "):]
        if not data:
            return

        objects = [obj.strip() for obj in data.split(',') if obj.strip()]

        for obj in objects:
            if '(' in obj:
                name = obj.split('(')[0].strip()
            else:
                name = obj
            action = self.get_action_for_object(name)
            if action:
                self.get_logger().info(f"Action: {action}")


    def get_action_for_object(self, object_name: str):
        # Quy định hành động theo tên object
        mapping = {
            'stop-sign': 'Stop the vehicle',
            'traffic-green': 'Go',
            'no-entry-road-sign': 'Do not enter',
            'crosswalk-sign': 'Slow down, watch pedestrians',
            'round-about-sign': 'Prepare to yield',
            'parking-sign': 'Parking area ahead',
            'stop-line': 'Stop at line',
            'one-way-road-sign': 'Follow one-way direction'
        }
        return mapping.get(object_name, None)

def main(args=None):
    rclpy.init(args=args)
    node = ActionDecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
