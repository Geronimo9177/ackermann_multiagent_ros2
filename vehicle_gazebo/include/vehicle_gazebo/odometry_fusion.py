#!/usr/bin/env python3
"""
OdometryFusion - Local + GPS fusion for smooth drift correction.

Logic:
  - Follows /odometry/local completely
  - Estimates position bias by comparing global vs local
  - Correction with hysteresis: activates when drift > threshold_on,
    deactivates when drift < threshold_off
  - Per-cycle saturation: correction never exceeds max_step meters
    per update, regardless of alpha tuning
  - Velocity guard: disables correction when vehicle is stationary
    to avoid drifting due to noisy GPS when stopped
"""

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry


class OdometryFusion(Node):

    def __init__(self):
        super().__init__('odometry_fusion')

        self.declare_parameter('threshold_on', 1) # [m] drift to activate correction 
        self.declare_parameter('threshold_off', 0.5) # [m] drift to deactivate correction
        self.declare_parameter('alpha', 0.02) # correction gain (0-1)
        self.declare_parameter('beta', 0.1) # EMA smoothing factor for residual (0-1)
        self.declare_parameter('max_step', 0.05) # [m] max correction per cycle 
        self.declare_parameter('velocity_threshold', 0.1) # [m/s] min speed to allow correction 

        self.threshold_on = self.get_parameter('threshold_on').value
        self.threshold_off = self.get_parameter('threshold_off').value
        self.alpha = self.get_parameter('alpha').value
        self.beta = self.get_parameter('beta').value
        self.max_step = self.get_parameter('max_step').value
        self.velocity_threshold = self.get_parameter('velocity_threshold').value

        if self.threshold_off >= self.threshold_on:
            self.get_logger().warn(
                f'threshold_off ({self.threshold_off}) >= threshold_on '
                f'({self.threshold_on}): invalid hysteresis, '
                f'using threshold_off = threshold_on * 0.5'
            )
            self.threshold_off = self.threshold_on * 0.5

        self.local_pose = None
        self.global_pose = None

        self.correction = np.zeros(2)   # [dx, dy] accumulated
        self.residual_ema = np.zeros(2)  # filtered residual
        self.correcting = False          # hysteresis state

        self.create_subscription(Odometry, '/odometry/local', self.local_cb, 10)
        self.create_subscription(Odometry, '/odometry/global', self.global_cb, 10)

        self.pub = self.create_publisher(Odometry, '/odometry/fused', 10)

        self.get_logger().info(
            f'OdometryFusion ready | '
            f'threshold=[{self.threshold_off:.2f}, {self.threshold_on:.2f}]m | '
            f'alpha={self.alpha:.3f} | beta={self.beta:.3f} | '
            f'max_step={self.max_step:.3f}m | '
            f'velocity_threshold={self.velocity_threshold:.3f}m/s'
        )

    # ========================================================================
    def local_cb(self, msg: Odometry):
        self.local_pose = msg
        self.publish_fused()

    # ========================================================================
    def global_cb(self, msg: Odometry):
        self.global_pose = msg

        if self.local_pose is None:
            return

        # Check if vehicle is moving; if stationary, skip correction
        # to avoid drifting due to GPS noise.
        vx = self.local_pose.twist.twist.linear.x
        vy = self.local_pose.twist.twist.linear.y
        speed = float(np.hypot(vx, vy))

        if speed < self.velocity_threshold:
            return

        # Current corrected position (local + accumulated correction)
        lx = self.local_pose.pose.pose.position.x + self.correction[0]
        ly = self.local_pose.pose.pose.position.y + self.correction[1]

        # Residual between global (GPS) and current corrected position
        gx = self.global_pose.pose.pose.position.x
        gy = self.global_pose.pose.pose.position.y

        residual = np.array([gx - lx, gy - ly])
        residual_norm = float(np.hypot(residual[0], residual[1]))

        # Hysteresis: activates when drift > threshold_on,
        # deactivates only when drift < threshold_off.
        if not self.correcting:
            if residual_norm > self.threshold_on:
                self.correcting = True
                self.get_logger().info(
                    f'Correction ACTIVATED | drift={residual_norm:.3f}m | speed={speed:.3f}m/s'
                )
        else:
            if residual_norm < self.threshold_off:
                self.correcting = False
                self.get_logger().info(
                    f'Correction DEACTIVATED | drift={residual_norm:.3f}m'
                )

        if not self.correcting:
            return

        # EMA of residual: filters instantaneous GPS noise before correction.
        self.residual_ema = ((1.0 - self.beta) * self.residual_ema
                             + self.beta * residual)

        # Correction with per-cycle saturation
        raw_dx = self.alpha * self.residual_ema[0]
        raw_dy = self.alpha * self.residual_ema[1]

        dx = float(np.clip(raw_dx, -self.max_step, self.max_step))
        dy = float(np.clip(raw_dy, -self.max_step, self.max_step))

        self.correction[0] += dx
        self.correction[1] += dy

        self.get_logger().info(
            f'drift={residual_norm:.3f}m | speed={speed:.3f}m/s | '
            f'ema=[{self.residual_ema[0]:.3f}, {self.residual_ema[1]:.3f}] | '
            f'step=[{dx:.4f}, {dy:.4f}] | '
            f'corr=[{self.correction[0]:.3f}, {self.correction[1]:.3f}]',
            throttle_duration_sec=1.0
        )

    # ========================================================================
    def publish_fused(self):
        if self.local_pose is None:
            return

        msg = Odometry()
        msg.header = self.local_pose.header
        msg.child_frame_id = self.local_pose.child_frame_id

        msg.pose.pose.position.x = (self.local_pose.pose.pose.position.x
                                    + self.correction[0])
        msg.pose.pose.position.y = (self.local_pose.pose.pose.position.y
                                    + self.correction[1])
        msg.pose.pose.position.z = self.local_pose.pose.pose.position.z
        msg.pose.pose.orientation = self.local_pose.pose.pose.orientation
        msg.pose.covariance = self.local_pose.pose.covariance
        msg.twist = self.local_pose.twist

        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(OdometryFusion())


if __name__ == '__main__':
    main()
