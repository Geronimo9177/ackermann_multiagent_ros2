#!/usr/bin/env python3
"""
Test-mode data logger.

Only meant to run with training_mode:=false. Records exactly one
evaluation episode: buffering starts on /sim/start (True) and stops on
/rl/result, at which point the buffered rows are written to a CSV file
and the node shuts itself down (which the launch file turns into a full
stack shutdown via an OnProcessExit handler).

Sampling is driven by /imu/data_raw (100 Hz), the fastest topic in the
stack. Slower topics (ground truth pose/twist, MPC debug, final command)
are cached and stamped onto each IMU-driven row.
"""
import csv
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, String, Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Imu

RESULT_NAMES = {0: 'RUNNING', 1: 'SUCCESS', 2: 'CRASH', 3: 'FALL'}

FIELDNAMES = [
    'stamp', 'trajectory_id',
    'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
    'vx_b', 'vy_b', 'vz', 'wx_gt', 'wy_gt', 'wz_gt',
    'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz',
    'x_ref', 'y_ref', 'yaw_ref', 'v_cmd_mpc', 'v_ref', 'steer_mpc',
    'track_progress', 'solve_ms',
    'v_final', 'w_final',
]


class TestLogger(Node):

    def __init__(self):
        super().__init__('test_logger')

        self.declare_parameter('output_dir', os.path.expanduser('~/mpc_test_logs'))
        self.declare_parameter('control_mode', 'nominal')

        self.output_dir = self.get_parameter('output_dir').value
        self.control_mode = self.get_parameter('control_mode').value
        os.makedirs(self.output_dir, exist_ok=True)

        self._recording      = False
        self._rows           = []
        self._trajectory_id  = ''

        # Latest cached state from the slower topics.
        self._gt  = None   # Odometry
        self._mpc = None   # Float64MultiArray.data
        self._cmd = None   # TwistStamped

        self.create_subscription(Bool,   '/sim/start',        self._start_cb, 1)
        self.create_subscription(Int32,  '/rl/result',        self._result_cb, 1)
        self.create_subscription(String, '/rl/trajectory_id', self._traj_cb, 1)
        self.create_subscription(Odometry, '/ground_truth_odom', self._gt_cb, 10)
        self.create_subscription(Float64MultiArray, '/mpc/debug', self._mpc_cb, 10)
        self.create_subscription(TwistStamped, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(Imu, '/imu/data_raw', self._imu_cb, 50)

        self.get_logger().info(
            f'test_logger ready (control_mode={self.control_mode}) — writing to {self.output_dir}')

    # ── Cheap caching callbacks ──────────────────────────────────────
    def _traj_cb(self, msg: String):
        self._trajectory_id = msg.data

    def _gt_cb(self, msg: Odometry):
        self._gt = msg

    def _mpc_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 11:
            self._mpc = msg.data

    def _cmd_cb(self, msg: TwistStamped):
        self._cmd = msg

    # ── Episode lifecycle ─────────────────────────────────────────────
    def _start_cb(self, msg: Bool):
        if not msg.data:
            return
        self._rows = []
        self._recording = True
        self.get_logger().info(f'[{self._trajectory_id or "?"}] Recording started')

    def _imu_cb(self, msg: Imu):
        if not self._recording:
            return
        gt, mpc, cmd = self._gt, self._mpc, self._cmd
        if gt is None or mpc is None or cmd is None:
            return  # slower topics haven't produced a sample yet this episode

        self._rows.append({
            'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'trajectory_id': self._trajectory_id,
            'x': gt.pose.pose.position.x,
            'y': gt.pose.pose.position.y,
            'z': gt.pose.pose.position.z,
            'qx': gt.pose.pose.orientation.x,
            'qy': gt.pose.pose.orientation.y,
            'qz': gt.pose.pose.orientation.z,
            'qw': gt.pose.pose.orientation.w,
            'vx_b': gt.twist.twist.linear.x,
            'vy_b': gt.twist.twist.linear.y,
            'vz': gt.twist.twist.linear.z,
            'wx_gt': gt.twist.twist.angular.x,
            'wy_gt': gt.twist.twist.angular.y,
            'wz_gt': gt.twist.twist.angular.z,
            'imu_ax': msg.linear_acceleration.x,
            'imu_ay': msg.linear_acceleration.y,
            'imu_az': msg.linear_acceleration.z,
            'imu_gx': msg.angular_velocity.x,
            'imu_gy': msg.angular_velocity.y,
            'imu_gz': msg.angular_velocity.z,
            'x_ref': mpc[3], 'y_ref': mpc[4], 'yaw_ref': mpc[5],
            'v_cmd_mpc': mpc[6], 'v_ref': mpc[7], 'steer_mpc': mpc[8],
            'track_progress': mpc[9], 'solve_ms': mpc[10],
            'v_final': cmd.twist.linear.x,
            'w_final': cmd.twist.angular.z,
        })

    def _result_cb(self, msg: Int32):
        if not self._recording:
            return
        self._recording = False

        result_name = RESULT_NAMES.get(msg.data, str(msg.data))
        traj_tag = self._trajectory_id.replace('/', '_') or 'unknown'
        path = os.path.join(
            self.output_dir,
            f'{self.control_mode}_{traj_tag}_{result_name}_{int(time.time())}.csv'
        )
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._rows)

        self.get_logger().info(
            f'Episode ended ({result_name}) — {len(self._rows)} samples written to {path}'
        )
        self.get_logger().info('Shutting down (single test episode complete).')
        rclpy.shutdown()


def main():
    rclpy.init()
    node = TestLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
