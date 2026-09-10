import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    run_debug_visualizer = LaunchConfiguration('run_debug_visualizer')
    training_mode = LaunchConfiguration('training_mode')
    control_mode         = LaunchConfiguration('control_mode')

    declare_training_mode = DeclareLaunchArgument(
        'training_mode', default_value='true',
        description='true = training; false = tiempo real')

    declare_control_mode = DeclareLaunchArgument(
        'control_mode',
        default_value='nominal',
        description="'nominal' = MPC puro | 'rule_based' = MPC + regla de frenado por cámara"
    )

    declare_run_debug_visualizer = DeclareLaunchArgument(
        'run_debug_visualizer',
        default_value='false',
        description='Run mpc_debug_visualizer when true',
    )

    webots_pkg = get_package_share_directory('vehicle_webots')
    controller_pkg = get_package_share_directory('vehicle_controller')

    webots_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(webots_pkg, 'launch', 'ackermann_webots.launch.py')
        ),
        launch_arguments={
            'training_mode': training_mode,
        }.items()
    )

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_pkg, 'launch', 'ackermann_controller.launch.py')
        ),
        launch_arguments={
            'control_mode': control_mode,
            'run_debug_visualizer': run_debug_visualizer,
        }.items()
    )

    episode_manager_node = Node(
        package='vehicle_controller',
        executable='episode_manager',
        name='episode_manager',
        output='screen',
        parameters=[
            {'training_mode': training_mode,
             'use_sim_time': True}
            ]
    )

    return LaunchDescription([
        declare_training_mode,
        declare_control_mode,
        declare_run_debug_visualizer,
        webots_launch,
        controller_launch,
        episode_manager_node
    ])