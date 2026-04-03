#!/usr/bin/env python3
"""
topp_node.py
------------
Receives nav_msgs/Path, computes a TOPP speed profile, and publishes
std_msgs/Float64MultiArray with 5 fields per point: [x, y, yaw, v, s].

Flat layout: [x0,y0,yaw0,v0,s0, x1,y1,yaw1,v1,s1, ...]
    layout.dim[0].size = N points
    layout.dim[1].size = 5 fields per point

Topics:
  Sub: /trajectory       (nav_msgs/Path)
  Pub: /trajectory_topp  (std_msgs/Float64MultiArray)
"""

import rclpy
from rclpy.node import Node

import numpy as np
from scipy.ndimage import gaussian_filter1d

from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from tf_transformations import euler_from_quaternion


class TOPP(Node):

    def __init__(self):
        super().__init__('topp')

        # Parameters
        # Vehicle parameters must match the MPC configuration.
        self.declare_parameter('v_max',       1.0)    # [m/s]  = MPC max_speed
        self.declare_parameter('at_max',      1.5)    # [m/s²] = MPC max_accel
        self.declare_parameter('max_steer',   0.5)    # [rad]  = MPC max_steer
        self.declare_parameter('wheelbase',   0.335)  # [m]    = MPC wheelbase L
        self.declare_parameter('kappa_sigma', 5.0)    # Gaussian smoothing for curvature
        self.declare_parameter('v_min',       0.05)   # [m/s]  Minimum speed

        self.v_max       = self.get_parameter('v_max').value
        self.at_max      = self.get_parameter('at_max').value
        self.max_steer   = self.get_parameter('max_steer').value
        self.wheelbase   = self.get_parameter('wheelbase').value
        self.kappa_sigma = self.get_parameter('kappa_sigma').value
        self.v_min       = self.get_parameter('v_min').value

        # ar_max is derived from vehicle geometry:
        #   ar = v² · κ = v² · tan(δ) / L
        #   ar_max = v_max² · tan(max_steer) / L
        # This ensures TOPP never requests a steering angle greater than max_steer.
        self.ar_max = (self.v_max**2 * np.tan(self.max_steer)) / self.wheelbase
        self.get_logger().info(
            f'Derived ar_max: {self.ar_max:.3f} m/s² '
            f'(v_max={self.v_max}, max_steer={np.degrees(self.max_steer):.1f}°, L={self.wheelbase})'
        )

        # Subscriber / publisher
        self.create_subscription(Path, '/trajectory', self.path_cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/trajectory_topp', 10)

        self.get_logger().info('TOPP node ready.')

    def path_cb(self, msg: Path):
        if len(msg.poses) < 3:
            self.get_logger().warn('Path too short for TOPP, skipping.')
            return

        x   = np.array([p.pose.position.x for p in msg.poses])
        y   = np.array([p.pose.position.y for p in msg.poses])
        yaw = np.array([self._yaw_from_pose(p) for p in msg.poses])

        s     = self._arclength(x, y)
        kappa = self._curvature(x, y, s)
        v     = self._velocity_profile(s, kappa)

        self._publish(x, y, yaw, v, s)

        self.get_logger().info(
            f'TOPP: {len(x)} pts | '
            f'v [{v.min():.2f}, {v.max():.2f}] m/s | '
            f'total dist: {s[-1]:.1f} m',
            throttle_duration_sec=2.0
        )

    def _arclength(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ds = np.hypot(np.diff(x), np.diff(y))
        return np.concatenate([[0.0], np.cumsum(ds)])

    def _curvature(self, x: np.ndarray, y: np.ndarray,
                   s: np.ndarray) -> np.ndarray:
        dx  = np.gradient(x,  s)
        dy  = np.gradient(y,  s)
        ddx = np.gradient(dx, s)
        ddy = np.gradient(dy, s)
        kappa = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-8)
        return gaussian_filter1d(kappa, sigma=self.kappa_sigma)

    def _velocity_profile(self, s: np.ndarray,
                          kappa: np.ndarray) -> np.ndarray:
        # Kinematic limit imposed by curvature
        v = np.minimum(self.v_max,
                       np.sqrt(self.ar_max / (np.abs(kappa) + 1e-6)))
        
        # Ensure the profile starts and ends at zero speed
        v[0] = 0.0
        v[-1] = 0.0

        # Forward pass: respect maximum acceleration
        for i in range(1, len(s)):
            ds   = s[i] - s[i - 1]
            v[i] = min(v[i], np.sqrt(v[i - 1]**2 + 2.0 * self.at_max * ds))

        # Backward pass: ensure the vehicle slows down before curves
        for i in range(len(s) - 2, -1, -1):
            ds   = s[i + 1] - s[i]
            v[i] = min(v[i], np.sqrt(v[i + 1]**2 + 2.0 * self.at_max * ds))

        v[1:-1] = np.maximum(v[1:-1], self.v_min)

        return v

    def _publish(self, x, y, yaw, v, s):
        n      = len(x)
        fields = 5  # x, y, yaw, v, s

        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='points', size=n,      stride=n * fields),
            MultiArrayDimension(label='fields', size=fields, stride=fields),
        ]
        # Interleave: [x0,y0,yaw0,v0,s0,  x1,y1,yaw1,v1,s1, ...]
        msg.data = np.column_stack([x, y, yaw, v, s]).flatten().tolist()
        self.pub.publish(msg)

    @staticmethod
    def _yaw_from_pose(pose_stamped) -> float:
        q = pose_stamped.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw


def main():
    rclpy.init()
    rclpy.spin(TOPP())


if __name__ == '__main__':
    main()