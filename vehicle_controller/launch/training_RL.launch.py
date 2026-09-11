import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

CHECKPOINT_DIR = os.path.join(os.path.expanduser('~'), 'ppo_checkpoints')
LOG_DIR        = os.path.join(os.path.expanduser('~'), 'ppo_test_logs')
RUN_ID_BASE    = 'speedbump'

def generate_launch_description():
    training_mode                = LaunchConfiguration('training_mode')
    seed                         = LaunchConfiguration('seed')
    checkpoint_dir                = LaunchConfiguration('checkpoint_dir')
    run_id                      = PythonExpression([
        "'", RUN_ID_BASE, "_seed_' + str(", seed, ")"
    ])

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
    declare_checkpoint_dir = DeclareLaunchArgument(
        'checkpoint_dir',
        default_value=CHECKPOINT_DIR,
        description='Directory ppo_agent loads/saves checkpoints from (picks the highest-episode file matching run_id)',
    )

    webots_pkg       = get_package_share_directory('vehicle_webots')
    controller_pkg   = get_package_share_directory('vehicle_controller')

    webots_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(webots_pkg, 'launch', 'ackermann_webots.launch.py')
        ),
        launch_arguments={
            'training_mode':    training_mode,
            'seed':             seed,
        }.items()
    )

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_pkg, 'launch', 'ackermann_controller.launch.py')
        ),
        launch_arguments={
            'run_debug_visualizer': 'false',
        }.items()
    )

    rl_master_node = Node(
        package='vehicle_controller',
        executable='rl_master',
        name='rl_master',
        output='screen',
        parameters=[{
            'training_mode':    training_mode,
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
            'run_id':           run_id,
            'checkpoint_dir':   checkpoint_dir,
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
            'checkpoint_dir': checkpoint_dir,
            'run_id':         run_id,
        }]
    )

    test_logger_node = Node(
        package='vehicle_controller',
        executable='test_logger',
        name='test_logger',
        output='screen',
        condition=UnlessCondition(training_mode),
        parameters=[{
            'output_dir': LOG_DIR,
            'run_id':     run_id,
            'seed':       seed,
        }]
    )

    shutdown_on_ppo_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=ppo_node,
            on_exit=[EmitEvent(event=Shutdown())],
        )
    )

    shutdown_on_test_logger_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=test_logger_node,
            on_exit=[EmitEvent(event=Shutdown())],
        )
    )

    return LaunchDescription([
        declare_training_mode,
        declare_seed,
        declare_checkpoint_dir,
        webots_launch,
        controller_launch,
        rl_master_node,
        ppo_node,
        ppo_debug_visualizer_node,
        test_logger_node,
        shutdown_on_ppo_exit,
        shutdown_on_test_logger_exit,
    ])