from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    run_debug_visualizer = LaunchConfiguration('run_debug_visualizer')

    declare_run_debug_visualizer = DeclareLaunchArgument(
        'run_debug_visualizer',
        default_value='false',
        description='Run mpc_debug_visualizer when true',
    )

    vehicle_config = PathJoinSubstitution([
        FindPackageShare('vehicle_controller'), 'config', ['tesla.yaml']
    ])

    return LaunchDescription([

        declare_run_debug_visualizer,

        Node(
            package='vehicle_controller',
            executable='ackermann_mpc',
            name='ackermann_mpc',
            output='screen',
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