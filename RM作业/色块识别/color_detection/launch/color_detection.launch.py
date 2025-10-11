from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera_id', default_value='0'),

        Node(
            package='color_detection',
            executable='camera_node',
            parameters=[{'camera_id': LaunchConfiguration('camera_id')}],
        ),

        Node(
            package='color_detection',
            executable='color_detect_node',
        ),
    ])

