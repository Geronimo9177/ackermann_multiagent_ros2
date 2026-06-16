import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Rutas de los paquetes
    webots_pkg = get_package_share_directory('vehicle_webots')
    controller_pkg = get_package_share_directory('vehicle_controller')

    webots_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(webots_pkg, 'launch', 'ackermann_webots.launch.py')
        ),
        launch_arguments={'training_mode': 'true'}.items()
    )

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_pkg, 'launch', 'ackermann_controller.launch.py')
        ),
        launch_arguments={'vehicle': 'tesla'}.items()
    )

    rl_master_node = Node(
        package='vehicle_controller',
        executable='rl_master',
        name='rl_master',
        output='screen',
        parameters=[{'training_mode': True}]
    )

    return LaunchDescription([
        webots_launch,
        controller_launch,
        rl_master_node
    ])