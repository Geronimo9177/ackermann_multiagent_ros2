from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg = 'vehicle_webots'
    pkg_share = get_package_share_directory(pkg)

    world_path = PathJoinSubstitution([
        FindPackageShare(pkg), 'worlds', 'vehicle_world.wbt'
    ])

    webots = WebotsLauncher(
        world=world_path,
        ros2_supervisor=True
    )

    return LaunchDescription([
        webots,
        webots._supervisor,

    ])