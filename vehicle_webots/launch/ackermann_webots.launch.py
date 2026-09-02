import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.wait_for_controller_connection import WaitForControllerConnection
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler, TimerAction, DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.conditions import UnlessCondition


def generate_launch_description():

    training_mode = LaunchConfiguration('training_mode')
    use_ground_truth = LaunchConfiguration('use_ground_truth')

    declare_training_mode = DeclareLaunchArgument(
        'training_mode',
        default_value='false',
        description='true = FAST + pause cada 50ms | false = tiempo real sin pausa'
    )

    declare_use_ground_truth = DeclareLaunchArgument(
        'use_ground_truth', default_value='false',
        description='Si es true, no lanza los EKF ni filtros'
    )

    pkg = 'vehicle_webots'
    package_dir = get_package_share_directory(pkg)

    set_extra_project_path = SetEnvironmentVariable(
        'WEBOTS_EXTRA_PROJECT_PATH', package_dir
    )

    robot_description_path = PathJoinSubstitution([
        FindPackageShare(pkg), 'config', 'tesla.urdf'
    ])

    sensor_fusion_config = PathJoinSubstitution([
        FindPackageShare(pkg), 'config', 'sensor_fusion.yaml'
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

    # TF estática base_link → gps_link
    gps_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['1.11', '0', '1.16', '0', '0', '0', 'base_link', 'gps_link'],
        output='screen',
        condition=UnlessCondition(use_ground_truth)
    )

    # Madgwick: fusiona IMU raw + magnetómetro → /imu/data
    imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter_madgwick',
        output='screen',
        parameters=[sensor_fusion_config],
        remappings=[
            ('imu/data_raw', '/imu/data_raw'),
            ('imu/mag',      '/magnetometer'),
            ('imu/data',     '/imu/data'),
        ],
        respawn=True,
        condition=UnlessCondition(use_ground_truth)
    )

    # EKF local: odom + imu/data → /odometry/local
    ekf_local = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_odom',
        output='screen',
        parameters=[sensor_fusion_config],
        remappings=[('odometry/filtered', '/odometry/local'),
                    ('set_pose', '/ekf_local/set_pose')],
        condition=UnlessCondition(use_ground_truth)
    )

    # EKF global: odometry/local + imu + GPS → /odometry/global
    ekf_global = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_map',
        output='screen',
        parameters=[sensor_fusion_config],
        remappings=[('odometry/filtered', '/odometry/global'),
                    ('set_pose', '/ekf_global/set_pose')],
        condition=UnlessCondition(use_ground_truth)
    )

    # navsat_transform: GPS + odometry/global → odometry/gps
    navsat = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_node',
        output='screen',
        parameters=[sensor_fusion_config],
        remappings=[
            ('imu/data',          '/imu/data'),
            ('gps/fix',           '/gps/fix'),
            ('odometry/filtered', '/odometry/global'),
            ('odometry/gps',      '/odometry/gps'),
        ],
        condition=UnlessCondition(use_ground_truth)
    )

    odom_fusion = Node(
        package='vehicle_webots',
        executable='odometry_fusion.py',
        name='odometry_fusion',
        output='screen',
        parameters=[{
            'drift_threshold': 1.5,
            'correction_alpha': 0.02,
        }],
        condition=UnlessCondition(use_ground_truth)
    )

    # Arrancar fusión sensorial cuando el driver esté listo
    waiting_nodes = WaitForControllerConnection(
        target_driver=vehicle_driver,
        nodes_to_start=[
            TimerAction(
                period=2.0, 
                actions=[
                    gps_tf,
                    imu_filter,
                    ekf_local,
                    ekf_global,
                    navsat,
                    odom_fusion,
                ]
            )
        ]
    )

    return LaunchDescription([
        set_extra_project_path,
        declare_training_mode,
        declare_use_ground_truth,
        webots,
        webots._supervisor,
        vehicle_driver,
        waiting_nodes,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])