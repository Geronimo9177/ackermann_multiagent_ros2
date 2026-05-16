import os
import launch
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    package_dir = get_package_share_directory('vehicle_webots')
    robot_description_path = os.path.join(package_dir, 'resource', 'tesla.urdf')

    webots = WebotsLauncher(
        world=os.path.join(package_dir, 'worlds', 'vehicle_world.wbt'),
        ros2_supervisor=True
    )

    vehicle_driver = WebotsController(
        robot_name='tesla',
        parameters=[
            {'robot_description': robot_description_path},
        ]
    )

    return LaunchDescription([
        webots,
        webots._supervisor,
        vehicle_driver,
        launch.actions.RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])