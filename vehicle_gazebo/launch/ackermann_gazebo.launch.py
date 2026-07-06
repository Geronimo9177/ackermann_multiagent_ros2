from launch import LaunchDescription
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit

def generate_launch_description():

    # Launch arguments
    use_sim_time = True

    declare_vehicle = DeclareLaunchArgument(
        'vehicle',
        default_value='prius',
        description='Vehicle model to simulate: traxxas or prius'
    )

    vehicle = LaunchConfiguration('vehicle')

    gz_args = LaunchConfiguration('gz_args', default='')

    # Paths
    gazebo_launch = PathJoinSubstitution([
        FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
    ])

    gazebo_world = PathJoinSubstitution([
        FindPackageShare('vehicle_gazebo'), 'worlds', 'baylands.sdf',
    ])

    robot_controller_config = PathJoinSubstitution([
        FindPackageShare('vehicle_gazebo'), 'config', 'ackermann_drive_controller.yaml',
    ])

    sensor_fusion_config = PathJoinSubstitution([
        FindPackageShare('vehicle_gazebo'), 'config', 'sensor_fusion.yaml',
    ])

    # Robot description: traxxas.urdf.xacro or prius.urdf.xacro
    robot_description = ParameterValue(
        Command(['xacro ', PathJoinSubstitution([
            FindPackageShare('vehicle_gazebo'), 'models', [vehicle, '.urdf.xacro']
        ])]),
        value_type=str
    )

    # Nodes

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
                '-name', 'ackermann_vehicle',
                '-topic', 'robot_description',
                '-allow_renaming', 'true',])

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster',
                   '--controller-manager-timeout', '60'],
    )

    ackermann_steering_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[        
            ['ackermann_steering_controller_', vehicle],
            '--param-file', robot_controller_config,
            '--controller-ros-args', ['-r /ackermann_steering_controller_', vehicle, '/tf_odometry:=/tf'],
            '--controller-ros-args', ['-r /ackermann_steering_controller_', vehicle, '/reference:=/cmd_vel'],
            '--controller-ros-args', ['-r /ackermann_steering_controller_', vehicle, '/odometry:=/ackermann_steering_controller/odometry_raw'],
        ],
    )

    return LaunchDescription([
        declare_vehicle,

        # Launch Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments=[('gz_args', [gz_args, ' -r -v 1 ', gazebo_world])]
        ),

        # Bridge Gazebo-ROS2
        Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[ 
                    '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                    "imu@sensor_msgs/msg/Imu@gz.msgs.IMU", # IMU data
                    '/ground_truth_odom@nav_msgs/msg/Odometry[gz.msgs.Odometry', # Ground truth odometry
                    '/gps/fix@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat',  # GPS
                    'magnetometer@sensor_msgs/msg/MagneticField@gz.msgs.Magnetometer',
                    # Segmentation camera
                    '/segmentation/colored_map@sensor_msgs/msg/Image[gz.msgs.Image', # Colored map
                    '/segmentation/labels_map@sensor_msgs/msg/Image[gz.msgs.Image', # Labels map
                    '/segmentation/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo', # Camera info
                
        ],
        remappings=[
            ('/imu', '/imu/data_raw'), 
            ('/gps/fix', '/gps/fix_raw'), 
        ],
        output='screen'
        ),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters= [{
                'robot_description': robot_description,
                'use_sim_time': use_sim_time,
            }],
            output='screen'
        ),

        # Spawn the robot
        gz_spawn_entity,

        # Spawn controllers after robot is spawned
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[
                    TimerAction(
                        period=5.0,  # Delay to ensure robot is fully spawned
                        actions=[joint_state_broadcaster_spawner]
                    )
                ],
            )
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[ackermann_steering_controller_spawner],
            )
        ),

        # Republish odometry with covariance
        Node(
            package='vehicle_gazebo',
            executable='odom_covariance_republisher.py',
            name='odom_covariance_republisher',
            output='screen'
        ),

        # Republish GPS with non-zero covariance for navsat_transform/global EKF
        Node(
            package='vehicle_gazebo',
            executable='gps_covariance_republisher.py',
            name='gps_covariance_republisher',
            output='screen',
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=ackermann_steering_controller_spawner,
                on_exit=[
                    Node(
                        package='imu_filter_madgwick',
                        executable='imu_filter_madgwick_node',
                        name='imu_filter_madgwick',
                        output='screen',
                        parameters=[sensor_fusion_config],
                        remappings=[
                            ('imu/data_raw', '/imu/data_raw'),
                            ('imu/mag',      '/magnetometer'),
                            ('imu/data',     '/imu/data')]
                    ),
                    
                    Node(
                        package='robot_localization',
                        executable='ekf_node',
                        name='ekf_filter_node_odom',
                        output='screen',
                        parameters=[sensor_fusion_config, {'use_sim_time': use_sim_time}],
                        remappings=[('odometry/filtered', '/odometry/local')]
                    ),
                    
                    Node(
                        package='robot_localization',
                        executable='ekf_node',
                        name='ekf_filter_node_map',
                        output='screen',
                        parameters=[sensor_fusion_config, {'use_sim_time': use_sim_time}],
                        remappings=[('odometry/filtered', '/odometry/global')]
                    ),

                    # navsat_transform (GPS - Odometry)
                    Node(
                        package='robot_localization',
                        executable='navsat_transform_node',
                        name='navsat_transform_node',
                        parameters=[sensor_fusion_config, {'use_sim_time': use_sim_time}],
                        remappings=[
                            ('imu', '/imu/data'),
                            ('gps/fix', '/gps/fix'),
                            ('odometry/filtered', '/odometry/global'),
                            ('odometry/gps', '/odometry/gps'),
                        ]
                    ),

                    Node(
                        package='vehicle_gazebo',
                        executable='odometry_fusion.py',
                        name='odometry_fusion',
                        output='screen',
                        parameters=[{
                            'drift_threshold':  1.5,   # empieza a corregir después de 1.5m de drift
                            'correction_alpha': 0.02,  # 2% por ciclo → corrección de 1m tarda ~1.5s
                        }]
                    ),
                ],
            )
        ),
        
    ])
