from launch import LaunchDescription
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit

def generate_launch_description():

    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)
    gz_args = LaunchConfiguration('gz_args', default='')

    # Gazebo launch file
    gazebo_launch = PathJoinSubstitution([
        FindPackageShare('ros_gz_sim'),
        'launch',
        'gz_sim.launch.py'
    ])

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
                '-name', 'ackermann_vehicle',
                '-topic', 'robot_description',
                '-allow_renaming', 'true',
                '-z', '0.5'])
    
    gazebo_world = PathJoinSubstitution(
        [
            FindPackageShare('vehicle_gazebo'),
            'worlds',
            'empty.sdf',
        ]
    )

    # Robot
    robot_description = ParameterValue(
        Command(['xacro ', PathJoinSubstitution([
                        FindPackageShare('vehicle_gazebo'),
                        'models',
                        'em_3905_base.urdf.xacro'
                    ])]),
        value_type=str
    )

    # Controllers
    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare('vehicle_gazebo'),
            'config',
            'ackermann_drive_controller.yaml',
        ]
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
    )
    ackermann_steering_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['ackermann_steering_controller',
                   '--param-file',
                   robot_controllers,
                   '--controller-ros-args',
                   '-r /ackermann_steering_controller/tf_odometry:=/tf',
                   '--controller-ros-args',
                   '-r /ackermann_steering_controller/reference:=/cmd_vel'
                   ],
    )

    #Extended Kalman Filter
    ekf_config = PathJoinSubstitution(
        [
            FindPackageShare('vehicle_gazebo'),
            'config',
            'ackermann_ekf.yaml',
        ]
    )

    return LaunchDescription([

        # Bridge 
        Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                   "imu@sensor_msgs/msg/Imu@gz.msgs.IMU", #IMU data
                   '/ground_truth_odom@nav_msgs/msg/Odometry[gz.msgs.Odometry', #Ground truth odometry
     ],
        output='screen'
        ),

        # Launch Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments=[('gz_args', [gz_args, ' -r -v 1 ', gazebo_world])]
            ),

        # Spawn controllers after robot is spawned
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[joint_state_broadcaster_spawner],
            )
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[ackermann_steering_controller_spawner],
            )
        ),

        # Spawn the robot
        gz_spawn_entity,
        
        # Launch Arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='If true, use simulated clock'),
        DeclareLaunchArgument(
            'description_format',
            default_value='urdf',
            description='Robot description format to use, urdf or sdf'),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters= [{
                'robot_description': robot_description
            }],
            output='screen'
        ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_config, {'use_sim_time': use_sim_time}],
            remappings=[('odometry/filtered', 'odometry/filtered')]
        )
    ])
