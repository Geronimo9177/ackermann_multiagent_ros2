#!/usr/bin/env python3
"""
DriftCorrector — fusión local + GPS para corrección suave de drift.

Lógica:
  - Sigue /odometry/local completamente (suave, sin saltos)
  - Cuando el drift acumulado vs /odometry/global supera `drift_threshold`,
    aplica una corrección suave (alpha por ciclo) hacia el GPS
  - El MPC usa /odometry/fused

Parámetros (ros2 param):
  drift_threshold  (m)   : drift mínimo para activar corrección  [default: 1.5]
  correction_alpha (0-1) : fracción de corrección por ciclo      [default: 0.02]
"""

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry


def wrap_angle(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class DriftCorrector(Node):

    def __init__(self):
        super().__init__('drift_corrector')

        self.declare_parameter('drift_threshold',  1.5)
        self.declare_parameter('correction_alpha', 0.02)

        self.drift_threshold  = self.get_parameter('drift_threshold').value
        self.correction_alpha = self.get_parameter('correction_alpha').value

        self.local_pose  = None
        self.global_pose = None

        # Corrección acumulada que se aplica encima del local
        self.correction = np.zeros(3)   # [dx, dy, dyaw]

        self.create_subscription(Odometry, '/odometry/local',
                                 self.local_cb,  10)
        self.create_subscription(Odometry, '/odometry/global',
                                 self.global_cb, 10)

        self.pub = self.create_publisher(Odometry, '/odometry/fused', 10)

        self.get_logger().info(
            f'DriftCorrector listo | '
            f'umbral={self.drift_threshold:.2f}m | '
            f'alpha={self.correction_alpha:.3f}'
        )

    # ------------------------------------------------------------------
    def local_cb(self, msg: Odometry):
        self.local_pose = msg
        self.publish_fused()

    # ------------------------------------------------------------------
    def global_cb(self, msg: Odometry):
        self.global_pose = msg

        if self.local_pose is None:
            return

        # Posición del local + corrección acumulada
        lx  = self.local_pose.pose.pose.position.x + self.correction[0]
        ly  = self.local_pose.pose.pose.position.y + self.correction[1]

        # Posición del global (GPS fusionado)
        gx  = self.global_pose.pose.pose.position.x
        gy  = self.global_pose.pose.pose.position.y

        drift_xy = float(np.hypot(gx - lx, gy - ly))

        if drift_xy > self.drift_threshold:
            # Corrección suave: mueve `alpha` hacia el GPS en este ciclo
            alpha = min(self.correction_alpha * (drift_xy / self.drift_threshold), 0.15)
            self.correction[0] += alpha * (gx - lx)
            self.correction[1] += alpha * (gy - ly)

            self.get_logger().info(
                f'Drift: {drift_xy:.3f} m | '
                f'corr: [{self.correction[0]:.3f}, {self.correction[1]:.3f}]',
                throttle_duration_sec=1.0
            )

    # ------------------------------------------------------------------
    def publish_fused(self):
        if self.local_pose is None:
            return

        msg = Odometry()
        msg.header          = self.local_pose.header
        msg.child_frame_id  = self.local_pose.child_frame_id

        # Posición: local + corrección acumulada
        msg.pose.pose.position.x = (self.local_pose.pose.pose.position.x
                                    + self.correction[0])
        msg.pose.pose.position.y = (self.local_pose.pose.pose.position.y
                                    + self.correction[1])
        msg.pose.pose.position.z  = self.local_pose.pose.pose.position.z

        # Orientación: siempre del local (suave, IMU+odometría)
        msg.pose.pose.orientation = self.local_pose.pose.pose.orientation

        # Covarianza de posición: la del local pero con Y/Z razonable
        # (el local diverge en Y porque no tiene ancla, aquí la fijamos)
        cov = list(self.local_pose.pose.covariance)
        cov[0]  = max(cov[0],  0.1)    # x  — mínimo razonable
        cov[7]  = 2.0                   # y  — fijo, ~1.4m de incertidumbre
        cov[14] = 4.0                   # z  — fijo
        msg.pose.covariance = cov

        # Twist siempre del local
        msg.twist = self.local_pose.twist

        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(DriftCorrector())


if __name__ == '__main__':
    main()