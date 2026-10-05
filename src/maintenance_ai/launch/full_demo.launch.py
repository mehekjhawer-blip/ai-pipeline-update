"""Gazebo robot + telemetry simulator + AI health node (+ optional CSV logger).

    ros2 launch maintenance_ai full_demo.launch.py
    ros2 launch maintenance_ai full_demo.launch.py log_csv:=true

Run the keyboard controller and the Streamlit dashboard in their own terminals.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup = os.path.join(get_package_share_directory('maintenance_robot'),
                           'launch', 'maintenance_bringup.launch.py')
    return LaunchDescription([
        DeclareLaunchArgument('log_csv', default_value='false',
                              description='also append telemetry to data/maintenance_data.csv'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(bringup)),
        Node(package='maintenance_monitor', executable='maintenance_sensor_sim', output='screen'),
        Node(package='maintenance_data_logger', executable='maintenance_data_logger_node',
             output='screen', condition=IfCondition(LaunchConfiguration('log_csv'))),
        Node(package='maintenance_ai', executable='health_node', output='screen'),
    ])
