import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

CHECKPOINT_DIR = os.path.join(os.path.expanduser('~'), 'ppo_checkpoints')
RUN_ID_BASE    = 'speedbump'

def generate_launch_description():
    use_ground_truth             = LaunchConfiguration('use_ground_truth')
    training_mode                = LaunchConfiguration('training_mode')
    seed                         = LaunchConfiguration('seed')
    run_id                      = PythonExpression([
        "'", RUN_ID_BASE, "_seed_' + str(", seed, ")"
    ])

    declare_use_ground_truth = DeclareLaunchArgument(
        'use_ground_truth',
        default_value='true',
    )
    declare_training_mode = DeclareLaunchArgument(
        'training_mode',
        default_value='true',
        description='true=train, false=evaluate last checkpoint',
    )
    declare_seed = DeclareLaunchArgument(
        'seed',
        default_value='0',
        description='Random seed shared by PPO, trajectory selection and Webots',
    )

    webots_pkg       = get_package_share_directory('vehicle_webots')
    controller_pkg   = get_package_share_directory('vehicle_controller')

    webots_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(webots_pkg, 'launch', 'ackermann_webots.launch.py')
        ),
        launch_arguments={
            'training_mode':    training_mode,
            'use_ground_truth': use_ground_truth,
            'seed':             seed,
        }.items()
    )

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_pkg, 'launch', 'ackermann_controller.launch.py')
        ),
        launch_arguments={
            'run_debug_visualizer': 'false',
            'vehicle':              'tesla',
            'use_ground_truth':      use_ground_truth
        }.items()
    )

    rl_master_node = Node(
        package='vehicle_controller',
        executable='rl_master',
        name='rl_master',
        output='screen',
        parameters=[{
            'training_mode':    training_mode,
            'use_ground_truth': use_ground_truth,
            'seed':             seed,
        }]
    )

    ppo_node = Node(
        package='vehicle_controller',
        executable='ppo_agent',
        name='ppo_agent',
        output='screen',
        parameters=[{
            'training_mode':    training_mode,
            'use_ground_truth': use_ground_truth,
            'run_id':           run_id,
            'checkpoint_dir':   CHECKPOINT_DIR,
            'seed':             seed,
        }]
    )

    ppo_debug_visualizer_node = Node(
        package='vehicle_controller',
        executable='ppo_debug_visualizer',
        name='ppo_debug_visualizer',
        output='screen',
        condition=IfCondition(training_mode),
        parameters=[{
            'checkpoint_dir': CHECKPOINT_DIR,
            'run_id':         run_id,
        }]
    )

    shutdown_on_ppo_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=ppo_node,
            on_exit=[EmitEvent(event=Shutdown())],
        )
    )

    return LaunchDescription([
        declare_use_ground_truth,
        declare_training_mode,
        declare_seed,
        webots_launch,
        controller_launch,
        rl_master_node,
        ppo_node,
        ppo_debug_visualizer_node,
        shutdown_on_ppo_exit,
    ])