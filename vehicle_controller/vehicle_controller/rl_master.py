#!/usr/bin/env python3
from tf_transformations import euler_from_quaternion
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool, Int32
from nav_msgs.msg import Odometry
import subprocess
import os
import glob
import random
import math
from collections import deque


class RLMaster(Node):

    # Episode end conditions
    FALL_Z_THRESHOLD   = -1.0   # m
    ROLLOVER_THRESHOLD = math.radians(60.0)

    STUCK_TICKS         = 500
    STUCK_PROGRESS_MIN  = 0.01 # %

    RESULT_RUNNING  = 0
    RESULT_SUCCESS  = 1
    RESULT_ROLLOVER = 2
    RESULT_STUCK    = 3
    RESULT_FALL     = 4
    def __init__(self):
        super().__init__('rl_master')

        self.declare_parameter('training_mode', True)
        self.declare_parameter('use_ground_truth', False)

        self.training_mode = self.get_parameter('training_mode').value
        self.use_ground_truth = self.get_parameter('use_ground_truth').value
        self.trajectories_subdir = 'train' if self.training_mode else 'test'

        self._traj_process   = None 

        self._odom_ready     = False
        self._episode_active = False
        self._start_timer    = None
        self._traj_timer     = None

        self._episode_count        = 0
        self._topp_ready           = False

        self._progress_history = deque(maxlen=self.STUCK_TICKS)

        pkg_path = get_package_share_directory('vehicle_controller')
        self.trajectories_dir = os.path.join(pkg_path, 'trajectories')

        # Publishers
        self.start_pub  = self.create_publisher(Bool,  '/sim/start',  1)
        self.reset_pub  = self.create_publisher(Bool,  '/sim/reset',  1)
        self.result_pub = self.create_publisher(Int32, '/rl/result', 1)

        # Subscribers
        self.topp_sub = self.create_subscription(
            Float64MultiArray, '/trajectory_topp', self._topp_cb, 1)
        
        if self.use_ground_truth:
            self.main_odom_topic = '/ground_truth_odom'
        else:
            self.main_odom_topic = '/odometry/fused'

        self.odom_sub = self.create_subscription(
            Odometry, self.main_odom_topic, self._main_odom_cb, 10) # Changed

        self.success_sub = self.create_subscription(
            Bool, '/mpc/success', self._success_cb, 1)

        self.gt_sub = self.create_subscription(
            Odometry, '/ground_truth_odom', self._gt_cb, 10)

        self.mpc_debug_sub = self.create_subscription(
            Float64MultiArray, '/mpc/debug', self._mpc_debug_cb, 10)
        
        self.get_logger().info('RLMaster node initialized')

    # ── Callbacks ────────────────────────────────────────────────
    def _success_cb(self, _msg: Bool):
        """The MPC reached the goal — successful episode."""
        if self._episode_active:
            self.get_logger().info('SUCCESS')
            self._end_episode(self.RESULT_SUCCESS)

    def _gt_cb(self, msg: Odometry):
        """Ground-truth velocity for crash detection."""
        if not self._episode_active:
            return

        z_pos = msg.pose.pose.position.z

        # Fall detection
        if z_pos < self.FALL_Z_THRESHOLD:
            self.get_logger().warn(f'FALL detected (z={z_pos:.2f}m)')
            self._end_episode(self.RESULT_FALL)
            return
        
        q = msg.pose.pose.orientation

        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if abs(roll) > self.ROLLOVER_THRESHOLD or abs(pitch) > self.ROLLOVER_THRESHOLD:
            self.get_logger().warn(
                f'ROLLOVER detected (roll={math.degrees(roll):.1f}°, pitch={math.degrees(pitch):.1f}°)'
            )
            self._end_episode(self.RESULT_ROLLOVER)
            return
        
    def _mpc_debug_cb(self, msg: Float64MultiArray):
        if not self._episode_active or len(msg.data) < 10:
            return

        track_progress = float(msg.data[9])
        self._progress_history.append(track_progress)

        if len(self._progress_history) == self._progress_history.maxlen:
            delta_prog  = track_progress - self._progress_history[0]

            if delta_prog < self.STUCK_PROGRESS_MIN:
                self.get_logger().warn(
                    f'STUCK detected ({delta_prog*100:.2f}% over {self.STUCK_TICKS} ticks)'
                )
                self._end_episode(self.RESULT_STUCK)

    def _main_odom_cb(self, msg: Odometry):
        if not self._odom_ready:
            self._odom_ready = True
            self.get_logger().info(f'{self.main_odom_topic} active — ready for a new episode')
            self._start_timer = self.create_timer(1.0, self._on_start_timer)
            return

    def _topp_cb(self, _msg):
        if self._episode_active or not self._episode_count:
            return 
        if self._topp_ready:
            return
        
        self._topp_ready = True
        self.get_logger().info('TOPP ready - starting episode in 0.5s')
        self._traj_timer = self.create_timer(0.5, self._on_traj_timer)
    
    def _kill_madgwick(self):
        if self.use_ground_truth:
            return
        
        try:
            subprocess.run(['pkill', '-f', 'imu_filter_madgwick_node'], check=False)
            self.get_logger().info('Madgwick filter terminated')
        except Exception as e:
            self.get_logger().error(f'Error while trying to terminate Madgwick: {e}')

    # ── Startup timers ───────────────────────────────────────────
    def _on_start_timer(self):
        if self._start_timer:
            self._start_timer.cancel()
            self._start_timer = None
        self.start_episode()

    def _on_traj_timer(self):
        if self._traj_timer:
            self._traj_timer.cancel()
            self._traj_timer = None
            
        msg = Bool()
        msg.data = True
        self.start_pub.publish(msg)
        self._episode_active = True
        self.get_logger().info(f'[Ep {self._episode_count}] activated')

    # ── Episode cycle ────────────────────────────────────────────
    def start_episode(self):
        traj_file = self._pick_random_trajectory()
        if traj_file is None:
            self.get_logger().error(f'No trajectories found in: {os.path.join(self.trajectories_dir, self.trajectories_subdir)}')
            return

        self._episode_count  += 1
        self._topp_ready      = False

        self._progress_history.clear()

        self.get_logger().info(f'[Ep {self._episode_count}] Trajectory: {traj_file}')

        # Immediate termination of the previous process to avoid freezes
        if self._traj_process is not None:
            try:
                self._traj_process.kill()
            except Exception:
                pass
            self._traj_process = None

        self._traj_process = subprocess.Popen([
            'ros2', 'run', 'vehicle_controller', 'trajectory_publisher',
            '--ros-args', '-p', f'trajectory_file:={traj_file}'
        ])
    
    def _end_episode(self, result: int):
        """Single exit point for an episode."""
        self._episode_active = False

        result_names = {
            self.RESULT_SUCCESS:  'SUCCESS',
            self.RESULT_ROLLOVER: 'ROLLOVER',
            self.RESULT_STUCK:    'STUCK',
            self.RESULT_FALL:     'FALL',
        }
        self.get_logger().info(
            f'[Ep {self._episode_count}] END -> {result_names.get(result, "?")}')

        # Publish result for the RL agent
        msg_r = Int32()
        msg_r.data = result
        self.result_pub.publish(msg_r)

        # Reset signal to Webots
        msg_b = Bool()
        msg_b.data = True
        self.reset_pub.publish(msg_b)
        self._kill_madgwick()

        self._odom_ready = False
        self.get_logger().info(f'Waiting for {self.main_odom_topic} for a new episode...')

    # ── Helpers ──────────────────────────────────────────────────
    def _pick_random_trajectory(self):
        files = glob.glob(os.path.join(self.trajectories_dir, self.trajectories_subdir, '*.csv'))
        if not files:
            return None
        return os.path.join(self.trajectories_subdir, os.path.basename(random.choice(files)))

    def shutdown(self):
        if self._traj_process is not None:
            try:
                self._traj_process.kill()
            except Exception:
                pass


def main():
    rclpy.init()
    node = RLMaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()