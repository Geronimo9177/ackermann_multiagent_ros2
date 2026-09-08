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
    seed = LaunchConfiguration('seed')

    declare_training_mode = DeclareLaunchArgument(
        'training_mode',
        default_value='false',
        description='true = FAST + pause cada 50ms | false = tiempo real sin pausa'
    )

    declare_seed = DeclareLaunchArgument(
        'seed', default_value='0',
        description='Random seed used by Webots controllers'
    )

    pkg = 'vehicle_webots'
    package_dir = get_package_share_directory(pkg)

    set_extra_project_path = SetEnvironmentVariable(
        'WEBOTS_EXTRA_PROJECT_PATH', package_dir
    )

    set_webots_seed = SetEnvironmentVariable(
        'WEBOTS_SEED', seed
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
        ]
    )

    return LaunchDescription([
        set_extra_project_path,
        set_webots_seed,
        declare_training_mode,
        declare_seed,
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