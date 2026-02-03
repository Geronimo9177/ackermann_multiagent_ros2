from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    ros_gz_sim_path = get_package_share_directory('ros_gz_sim')

    gazebo_launch = PathJoinSubstitution([
        ros_gz_sim_path,
        'launch',
        'gz_sim.launch.py'
    ])

    robot_description = ParameterValue(
        Command(['xacro ', PathJoinSubstitution([
                        FindPackageShare('vehicle_gazebo'),
                        'models',
                        'em_3905_base.urdf.xacro'
                    ])]),
        value_type=str
    )


    return LaunchDescription([

        # Gazebo resource paths (URDF, meshes, worlds)
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            PathJoinSubstitution([
                FindPackageShare('vehicle_gazebo')
            ])
        ),

        # Launch Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments={
                'gz_args': '-r empty.sdf',
                'on_exit_shutdown': 'true'
            }.items(),
        ),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters= [{
                'robot_description': robot_description
            }],
            output='screen'
        ),

        # SPAWN DEL ROBOT EN GAZEBO
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'ackermann_vehicle',
                '-topic', 'robot_description',
                '-z', '0.5'
            ],
            output='screen'
        ),
    ])
