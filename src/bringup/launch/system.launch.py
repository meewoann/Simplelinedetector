from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    camera_node = Node(
        package='camera_reader',
        executable='camera_reader_node',
        name='camera_reader',
        output='screen'
    )

    lane_node = Node(
        package='lane_detect',
        executable='lane_detect_node',
        name='lane_detect',
        output='screen'
    )

    # object_node = Node(
    #     package='object_detection',
    #     executable='object_node',
    #     name='object_detection',
    #     output='screen'
    # )

    # decision_node = Node(
    #     package='decision_making',
    #     executable='decision_node',
    #     name='decision_making',
    #     output='screen'
    # )

    return LaunchDescription([
        camera_node,
        lane_node,
        # object_node,
        # decision_node
    ])
