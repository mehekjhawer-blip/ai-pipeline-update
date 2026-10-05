"""Only the AI health node (use when the robot + sensor sim are already running)."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='maintenance_ai', executable='health_node', output='screen'),
    ])
