#!/usr/bin/env python3
"""
topp_node.py
------------
Receives nav_msgs/Path with NURBS control points,
reconstructs the path using CubicSpline (lines + curves),
computes a TOPP speed profile, and publishes
std_msgs/Float64MultiArray with 5 fields per point: [x, y, yaw, v, s].

Topics:
  Sub: /trajectory       (nav_msgs/Path)
  Pub: /trajectory_topp  (std_msgs/Float64MultiArray)
"""

import rclpy
from rclpy.node import Node

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


class TOPP(Node):

    def __init__(self):
        super().__init__('topp')

        # Vehicle parameters
        self.declare_parameter('v_max',                8.0)
        self.declare_parameter('at_max',               2.5)
        self.declare_parameter('bt_max',               5.0)
        self.declare_parameter('mu',                   0.9)
        self.declare_parameter('lat_usage',            0.6)
        self.declare_parameter('max_steer',            0.6458)
        self.declare_parameter('wheelbase',            2.55)
        self.declare_parameter('v_min',                0.5)
        # Reconstruction parameters
        self.declare_parameter('jump_threshold_factor', 5.0)
        self.declare_parameter('samples_per_meter',     4.0)
        self.declare_parameter('curve_samples',         50)
        self.declare_parameter('yaw_sigma',             3.0)

        self.v_max       = self.get_parameter('v_max').value
        self.at_max      = self.get_parameter('at_max').value
        self.bt_max      = self.get_parameter('bt_max').value
        self.mu          = self.get_parameter('mu').value
        self.lat_usage   = self.get_parameter('lat_usage').value
        self.max_steer   = self.get_parameter('max_steer').value
        self.wheelbase   = self.get_parameter('wheelbase').value
        self.v_min       = self.get_parameter('v_min').value

        self.jump_factor   = self.get_parameter('jump_threshold_factor').value
        self.spm           = self.get_parameter('samples_per_meter').value
        self.curve_samples = self.get_parameter('curve_samples').value
        self.yaw_sigma     = self.get_parameter('yaw_sigma').value

        self.ar_max = self.mu * 9.81 * self.lat_usage
        self.get_logger().info(
            f'ar_max={self.ar_max:.3f} m/s² | '
            f'at_max={self.at_max:.2f} | bt_max={self.bt_max:.2f}'
        )

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.create_subscription(Path, '/trajectory', self.path_cb, qos)
        self.pub = self.create_publisher(
            Float64MultiArray, '/trajectory_topp', 10)

        self.get_logger().info('TOPP node ready.')

    # ─────────────────────────────────────────────────────────
    # Callback
    # ─────────────────────────────────────────────────────────
    def path_cb(self, msg: Path):
        if len(msg.poses) < 3:
            self.get_logger().warn('Path too short, skipping.')
            return

        # Extract control points
        x_ctrl = np.array([p.pose.position.x for p in msg.poses])
        y_ctrl = np.array([p.pose.position.y for p in msg.poses])

        # Reconstruct full path
        x, y, kappa, s = self._reconstruct(x_ctrl, y_ctrl)

        # Compute yaw from reconstructed path
        yaw = self._compute_yaw(x, y, s)

        # Velocity profile
        v = self._velocity_profile(s, kappa)

        self._publish(x, y, yaw, v, s)

        self.get_logger().info(
            f'TOPP: {len(x)} pts | '
            f'v [{v.min():.2f}, {v.max():.2f}] m/s | '
            f'dist: {s[-1]:.1f} m | '
            f'R_min: {1/(np.max(np.abs(kappa))+1e-8):.2f} m',
            throttle_duration_sec=2.0
        )

    # ─────────────────────────────────────────────────────────
    # Path reconstruction — same logic as visualizer
    # ─────────────────────────────────────────────────────────
    def _reconstruct(self, x_ctrl, y_ctrl):
        ds        = np.hypot(np.diff(x_ctrl), np.diff(y_ctrl))
        median_ds = np.median(ds)
        threshold = median_ds * self.jump_factor

        # Detect segments
        segments = []
        i = 0
        while i < len(x_ctrl) - 1:
            if ds[i] > threshold:
                segments.append(('line', [i, i + 1]))
                i += 1
            else:
                j = i
                while j < len(x_ctrl) - 1 and ds[j] <= threshold:
                    j += 1
                segments.append(('curve', list(range(i, j + 1))))
                i = j

        self.get_logger().info(
            f'Control points: {len(x_ctrl)} | '
            f'Segments: {len(segments)} | '
            f'Median spacing: {median_ds:.2f} m'
        )

        # Reconstruct each segment
        all_x, all_y, all_k = [], [], []

        for seg_idx, (seg_type, idx) in enumerate(segments):
            xi      = x_ctrl[idx]
            yi      = y_ctrl[idx]
            is_last = (seg_idx == len(segments) - 1)

            if seg_type == 'line':
                length = np.hypot(xi[-1] - xi[0], yi[-1] - yi[0])
                n      = max(2, int(length * self.spm))
                t      = np.linspace(0, 1, n, endpoint=is_last)
                seg_x  = xi[0] + t * (xi[-1] - xi[0])
                seg_y  = yi[0] + t * (yi[-1] - yi[0])
                seg_k  = np.zeros(n)

            else:
                ds_loc = np.hypot(np.diff(xi), np.diff(yi))
                s_loc  = np.concatenate([[0], np.cumsum(ds_loc)])
                cs_x   = CubicSpline(s_loc, xi)
                cs_y   = CubicSpline(s_loc, yi)
                n      = len(idx) * self.curve_samples
                s_fine = np.linspace(s_loc[0], s_loc[-1], n,
                                     endpoint=is_last)
                seg_x  = cs_x(s_fine)
                seg_y  = cs_y(s_fine)
                dx     = cs_x(s_fine, 1)
                dy     = cs_y(s_fine, 1)
                ddx    = cs_x(s_fine, 2)
                ddy    = cs_y(s_fine, 2)
                seg_k  = ((dx * ddy - dy * ddx) /
                          ((dx**2 + dy**2)**1.5 + 1e-8))

            all_x.extend(seg_x)
            all_y.extend(seg_y)
            all_k.extend(seg_k)

        x     = np.array(all_x)
        y     = np.array(all_y)
        kappa = np.array(all_k)

        # Remove residual duplicates
        mask  = np.concatenate(
            [[True], np.hypot(np.diff(x), np.diff(y)) > 1e-4])
        x     = x[mask]
        y     = y[mask]
        kappa = kappa[mask]

        # Arclength
        ds_f = np.hypot(np.diff(x), np.diff(y))
        s    = np.concatenate([[0], np.cumsum(ds_f)])

        return x, y, kappa, s

    # ─────────────────────────────────────────────────────────
    # Yaw
    # ─────────────────────────────────────────────────────────
    def _compute_yaw(self, x, y, s):
        dx_s = gaussian_filter1d(np.gradient(x, s), sigma=self.yaw_sigma)
        dy_s = gaussian_filter1d(np.gradient(y, s), sigma=self.yaw_sigma)
        return np.arctan2(dy_s, dx_s)

    # ─────────────────────────────────────────────────────────
    # Friction ellipse
    # ─────────────────────────────────────────────────────────
    def _longitudinal_limit(self, v, kappa, cap):
        a_lat = (v ** 2) * abs(kappa)
        if a_lat >= self.ar_max:
            return 0.0
        ratio = a_lat / self.ar_max
        return float(cap * np.sqrt(max(0.0, 1.0 - ratio**2)))

    # ─────────────────────────────────────────────────────────
    # Velocity profile
    # ─────────────────────────────────────────────────────────
    def _velocity_profile(self, s, kappa):
        kappa_safe = np.maximum(np.abs(kappa), 1e-4)
        v = np.minimum(self.v_max, np.sqrt(self.ar_max / kappa_safe))

        v[0]  = min(v[0], self.v_min)
        v[-1] = 0.0

        for _ in range(2):
            # Forward pass
            for i in range(1, len(s)):
                ds    = max(s[i] - s[i-1], 1e-6)
                a_acc = self._longitudinal_limit(v[i-1], kappa[i-1], self.at_max)
                v[i]  = min(v[i], np.sqrt(v[i-1]**2 + 2.0 * a_acc * ds))

            # Backward pass
            for i in range(len(s) - 2, -1, -1):
                ds      = max(s[i+1] - s[i], 1e-6)
                v_ref   = max(v[i], v[i+1])
                k_ref   = max(abs(kappa[i]), abs(kappa[i+1]))
                a_brake = self._longitudinal_limit(v_ref, k_ref, self.bt_max)
                v[i]    = min(v[i], np.sqrt(v[i+1]**2 + 2.0 * a_brake * ds))

        return np.minimum(v, self.v_max)

    # ─────────────────────────────────────────────────────────
    # Publish
    # ─────────────────────────────────────────────────────────
    def _publish(self, x, y, yaw, v, s):
        n      = len(x)
        fields = 5

        msg            = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='points', size=n,
                                stride=n * fields),
            MultiArrayDimension(label='fields', size=fields,
                                stride=fields),
        ]
        msg.data = np.column_stack([x, y, yaw, v, s]).flatten().tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(TOPP())


if __name__ == '__main__':
    main()