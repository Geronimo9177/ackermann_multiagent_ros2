from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():

    xacro_file = PathJoinSubstitution([
        FindPackageShare('ackermann_vehicle_description'),
        'urdf',
        'em_3905_base.urdf.xacro'
    ])


    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )


    return LaunchDescription([

        # Publica TF desde el URDF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description
            }]
        ),

        # Publica joint_states "vacíos"
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui'
        ),


        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        )
    ])
