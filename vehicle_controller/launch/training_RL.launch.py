import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    use_ground_truth = LaunchConfiguration('use_ground_truth')
    run_ppo_debug_visualizer = LaunchConfiguration('run_ppo_debug_visualizer')
    
    declare_use_ground_truth = DeclareLaunchArgument(
        'use_ground_truth',
        default_value='true', 
    )

    declare_run_ppo_debug_visualizer = DeclareLaunchArgument(
        'run_ppo_debug_visualizer',
        default_value='true',
    )

    # Rutas de los paquetes
    webots_pkg = get_package_share_directory('vehicle_webots')
    controller_pkg = get_package_share_directory('vehicle_controller')

    webots_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(webots_pkg, 'launch', 'ackermann_webots.launch.py')
        ),
        launch_arguments={
            'training_mode': 'true',
            'use_ground_truth': use_ground_truth
        }.items()
    )

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_pkg, 'launch', 'ackermann_controller.launch.py')
        ),
        launch_arguments={
            'run_debug_visualizer': 'false',
            'vehicle': 'tesla',
            'use_ground_truth': use_ground_truth
        }.items()
    )

    rl_master_node = Node(
        package='vehicle_controller',
        executable='rl_master',
        name='rl_master',
        output='screen',
        parameters=[
            {'training_mode': True,
            'use_ground_truth': use_ground_truth}
            ]
    )

    ppo_node = Node(
        package='vehicle_controller',
        executable='ppo_agent',
        name='ppo_agent',
        output='screen',
        parameters=[{
            'training_mode':    True,
            'use_ground_truth': use_ground_truth,
            'run_id':           'speedbump_v1',
            'checkpoint_dir':   os.path.join(os.path.expanduser('~'), 'ppo_checkpoints'),
        }]
    )

    ppo_debug_visualizer_node = Node(
        package='vehicle_controller',
        executable='ppo_debug_visualizer',
        name='ppo_debug_visualizer',
        output='screen',
        condition=IfCondition(run_ppo_debug_visualizer)
    )

    return LaunchDescription([
        declare_use_ground_truth,
        declare_run_ppo_debug_visualizer,
        webots_launch,
        controller_launch,
        rl_master_node,
        ppo_node,
        ppo_debug_visualizer_node,
    ])