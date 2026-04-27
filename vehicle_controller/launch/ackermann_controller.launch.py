from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    run_debug_visualizer = LaunchConfiguration('run_debug_visualizer')

    declare_run_debug_visualizer = DeclareLaunchArgument(
        'run_debug_visualizer',
        default_value='false',
        description='Run mpc_debug_visualizer when true',
    )

    return LaunchDescription([

        declare_run_debug_visualizer,

        # Ackermann MPC node
        Node(
            package='vehicle_controller',
            executable='ackermann_mpc',
            name='ackermann_mpc',
            output='screen',
        ),

        # TOPP (Time-Optimal Path Parametrization) node
        Node(
            package='vehicle_controller',
            executable='topp',
            name='topp',
            output='screen',
        ),

        # MPC Debug Visualizer
        Node(
            package='vehicle_controller',
            executable='mpc_debug_visualizer',
            name='mpc_debug_visualizer',
            output='screen',
            condition=IfCondition(run_debug_visualizer),
        ),
    ])