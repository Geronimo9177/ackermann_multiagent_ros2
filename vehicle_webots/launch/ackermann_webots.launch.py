import os
import launch
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler, DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    training_mode = LaunchConfiguration('training_mode')
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_training_mode = DeclareLaunchArgument(
        'training_mode',
        default_value='false',
        description='true = tiempo real durante el calculo y FAST despues | false = tiempo real'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Sincroniza todos los nodos con /clock de Webots (Ros2Supervisor)'
    )

    pkg = 'vehicle_webots'
    package_dir = get_package_share_directory(pkg)

    set_extra_project_path = SetEnvironmentVariable(
        'WEBOTS_EXTRA_PROJECT_PATH', package_dir
    )

    robot_description_path = PathJoinSubstitution([
        FindPackageShare(pkg), 'config', 'tesla.urdf'
    ])

    webots = WebotsLauncher(
        world=os.path.join(package_dir, 'worlds', 'baylands.wbt'),
        ros2_supervisor=True
    )

    vehicle_driver = WebotsController(
        robot_name='tesla',
        parameters=[
            {'robot_description': robot_description_path},
            {'training_mode': training_mode},
            {'use_sim_time': use_sim_time},
        ]
    )

    return LaunchDescription([
        set_extra_project_path,
        declare_training_mode,
        declare_use_sim_time,
        webots,
        webots._supervisor,
        vehicle_driver,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])