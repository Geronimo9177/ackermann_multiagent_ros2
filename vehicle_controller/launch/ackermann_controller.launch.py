import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    run_debug_visualizer = LaunchConfiguration('run_debug_visualizer')
    vehicle = LaunchConfiguration('vehicle')
    use_ground_truth = LaunchConfiguration('use_ground_truth')

    declare_run_debug_visualizer = DeclareLaunchArgument(
        'run_debug_visualizer',
        default_value='false',
        description='Run mpc_debug_visualizer when true',
    )

    declare_vehicle = DeclareLaunchArgument(
        'vehicle',
        default_value='tesla',
        description='Vehicle model: toyota or tesla'
    )

    declare_use_ground_truth = DeclareLaunchArgument(
        'use_ground_truth', 
        default_value='false',
        description='Sobreescribe el YAML para decirle al MPC qué odometría usar'
    )

    vehicle_config = PathJoinSubstitution([
        FindPackageShare('vehicle_controller'), 'config', [vehicle, '.yaml']
    ])

    return LaunchDescription([

        declare_run_debug_visualizer,
        declare_vehicle,
        declare_use_ground_truth,

        Node(
            package='vehicle_controller',
            executable='ackermann_mpc',
            name='ackermann_mpc',
            output='screen',
            parameters=[vehicle_config, {'use_ground_truth': use_ground_truth}],
        ),

        Node(
            package='vehicle_controller',
            executable='topp',
            name='topp',
            output='screen',
            parameters=[vehicle_config],
        ),

        Node(
            package='vehicle_controller',
            executable='mpc_debug_visualizer',
            name='mpc_debug_visualizer',
            output='screen',
            condition=IfCondition(run_debug_visualizer),
        ),
    ])