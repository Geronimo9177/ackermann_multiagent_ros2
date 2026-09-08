#!/usr/bin/env python3
from tf_transformations import euler_from_quaternion
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool, Int32
from std_srvs.srv import Trigger
from nav_msgs.msg import Odometry
import subprocess
import os
import glob
import random
import math


class RLMaster(Node):

    # ── Episode termination conditions ──────────────────────────
    CRASH_SPEED_THRESHOLD  = 0.1   # m/s
    CRASH_SPEED_DURATION   = 0.5    # s
    FALL_Z_THRESHOLD       = -1.0   # m
    MIN_INITIAL_SPEED      = 0.3    # m/s 
    ROLLOVER_THRESHOLD     = math.radians(60.0)

    RESULT_RUNNING  = 0
    RESULT_SUCCESS  = 1
    RESULT_CRASH    = 2
    RESULT_FALL     = 3

    def __init__(self):
        super().__init__('rl_master')

        self.declare_parameter('training_mode', True)

        self.training_mode = self.get_parameter('training_mode').value
        self.trajectories_subdir = 'train' if self.training_mode else 'test'

        # Only manage the trajectory process
        self._traj_process   = None 

        self._odom_ready    = False
        self._episode_active = False
        self._start_timer    = None
        self._traj_timer     = None

        # State used for episode termination detection
        self._low_speed_since      = None
        self._vehicle_moved        = False
        self._episode_count        = 0
        self._topp_ready           = False
        self._gt_speed             = 0.0
        
        pkg_path = get_package_share_directory('vehicle_controller')
        self.trajectories_dir = os.path.join(pkg_path, 'trajectories')

        # Publishers
        self.start_pub  = self.create_publisher(Bool,  '/sim/start',  1)
        self.reset_pub  = self.create_publisher(Bool,  '/sim/reset',  1)
        self.result_pub = self.create_publisher(Int32, '/rl/result', 1)
        self.speedbump_client = self.create_client(
            Trigger, '/speedbump_control/reset_episode')

        # Subscribers
        self.topp_sub = self.create_subscription(
            Float64MultiArray, '/trajectory_topp', self._topp_cb, 1)

        self.success_sub = self.create_subscription(
            Bool, '/mpc/success', self._success_cb, 1)

        self.gt_sub = self.create_subscription(
            Odometry, '/ground_truth_odom', self._gt_cb, 10)
        
        self.get_logger().info('RLMaster started')

    # ── Callbacks ────────────────────────────────────────────────
    def _success_cb(self, _msg: Bool):
        """The MPC reached the goal - successful episode."""
        if self._episode_active:
            self.get_logger().info('SUCCESS - vehicle reached the destination')
            self._end_episode(self.RESULT_SUCCESS)

    def _gt_cb(self, msg: Odometry):
        if not self._odom_ready:
                    self._odom_ready = True
                    self.get_logger().info('Active - starting...')
                    self._start_timer = self.create_timer(1.0, self._on_start_timer)
                    return
        
        """Ground-truth velocity used for crash detection."""
        if not self._episode_active:
            return

        z_pos = msg.pose.pose.position.z

        # Vehicle fell off the map
        if z_pos < self.FALL_Z_THRESHOLD:
            self.get_logger().warn(f'FALL detected (z={z_pos:.2f} m)')
            self._end_episode(self.RESULT_FALL)
            return
        
        q = msg.pose.pose.orientation

        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if abs(roll) > self.ROLLOVER_THRESHOLD or abs(pitch) > self.ROLLOVER_THRESHOLD:
            self.get_logger().warn(
                f'ROLLOVER detected (roll={math.degrees(roll):.1f} deg, pitch={math.degrees(pitch):.1f} deg)'
            )
            self._end_episode(self.RESULT_CRASH)
            return
        
        now = self.get_clock().now().nanoseconds * 1e-9

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.sqrt(vx**2 + vy**2)

        if speed > self.MIN_INITIAL_SPEED:
            self._vehicle_moved = True

        if self._vehicle_moved:
            if speed < self.CRASH_SPEED_THRESHOLD:
                if self._low_speed_since is None:
                    self._low_speed_since = now
                elif now - self._low_speed_since >= self.CRASH_SPEED_DURATION:
                    self.get_logger().warn(f'CRASH detected (v={speed:.3f} m/s)')
                    self._end_episode(self.RESULT_CRASH)
            else:
                self._low_speed_since = None

    def _topp_cb(self, _msg):
        if self._episode_active or not self._episode_count:
            return 
        if self._topp_ready:
            return
        
        self._topp_ready = True
        self.get_logger().info('TOPP ready - starting episode in 0.5 s...')
        self._traj_timer = self.create_timer(0.5, self._on_traj_timer)

    # ── Startup timers ──────────────────────────────────────────
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
        self.get_logger().info(f'[Ep {self._episode_count}] Active - /rl/start published')

    # ── Episode cycle ───────────────────────────────────────────
    def start_episode(self):
        traj_file = self._pick_random_trajectory()
        if traj_file is None:
            self.get_logger().error(
                f'No trajectories found in: {os.path.join(self.trajectories_dir, self.trajectories_subdir)}'
            )
            return

        self._episode_count += 1
        self._low_speed_since = None
        self._vehicle_moved   = False
        self._topp_ready      = False
        
        self.get_logger().info(
            f'[Ep {self._episode_count}] Resetting speed bumps before: {traj_file}')

        if not self.speedbump_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                'SpeedBumpSupervisor service is not available; episode not started')
            return

        future = self.speedbump_client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda result: self._on_speedbumps_reset(result, traj_file))

    def _on_speedbumps_reset(self, future, traj_file):
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(f'Speed bump reset failed: {error}')
            return

        if not response.success:
            self.get_logger().error(
                f'Speed bump reset rejected: {response.message}')
            return

        self.get_logger().info(
            f'[Ep {self._episode_count}] {response.message}')

        # Immediately terminate the previous process to prevent freezes
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
            self.RESULT_SUCCESS: 'SUCCESS',
            self.RESULT_CRASH:   'CRASH',
            self.RESULT_FALL:    'FALL',
        }
        self.get_logger().info(
            f'[Ep {self._episode_count}] FIN → {result_names.get(result, "?")}')

        # Publish the result for the RL agent
        msg_r = Int32()
        msg_r.data = result
        self.result_pub.publish(msg_r)

        # Send a reset signal to Webots
        msg_b = Bool()
        msg_b.data = True
        self.reset_pub.publish(msg_b)

        self._odom_ready = False
        self.get_logger().info('Waiting for the next episode...')

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