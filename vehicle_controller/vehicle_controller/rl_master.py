#!/usr/bin/env python3
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
import subprocess
import os
import glob
import random
import time


class RLMaster(Node):
    def __init__(self):
        super().__init__('rl_master')

        self.declare_parameter('training_mode', True)

        self.training_mode    = self.get_parameter('training_mode').value

        self._processes      = []
        self._fused_ready    = False
        self._start_timer    = None
        self._episode_active = False
        pkg_path = get_package_share_directory('vehicle_controller')
        self.trajectories_dir = os.path.join(pkg_path, 'trajectories')

        # Publishers
        self.start_pub = self.create_publisher(Bool, '/rl/start', 1)
        self.reset_pub = self.create_publisher(Bool, '/rl/reset', 1)

        # Esperar odometry/fused como señal de que el stack está listo
        self.fused_sub = self.create_subscription(
            Odometry, '/odometry/fused', self._fused_cb, 1)

        self._launch_stack()
        self.get_logger().info('RLMaster iniciado, esperando odometry/fused...')

    # ─────────────────────────────────────────────────────────────
    def _launch_stack(self):
        training_str = 'true' if self.training_mode else 'false'

        p1 = subprocess.Popen([
            'ros2', 'launch', 'vehicle_webots',
            'ackermann_webots.launch.py',
            f'training_mode:={training_str}'
        ])
        self._processes.append(p1)

        p2 = subprocess.Popen([
            'ros2', 'launch', 'vehicle_controller',
            'ackermann_controller.launch.py',
            'vehicle:=tesla'
        ])
        self._processes.append(p2)

    # ─────────────────────────────────────────────────────────────
    def _fused_cb(self, _msg):
        if self._fused_ready:
            return
        self._fused_ready = True
        self.get_logger().info('odometry/fused activo — arrancando episodio en 2s...')

        # Timer de un solo disparo: arranca episodio tras 2s
        self._start_timer = self.create_timer(2.0, self._on_start_timer)

    def _on_start_timer(self):
        self._start_timer.cancel()
        self._start_timer = None
        self.start_episode()

    # ─────────────────────────────────────────────────────────────
    def start_episode(self):
        traj_file = self._pick_random_trajectory()
        if traj_file is None:
            self.get_logger().error(
                f'No hay trayectorias en: {self.trajectories_dir}')
            return

        filename = os.path.basename(traj_file)
        self.get_logger().info(f'Trayectoria elegida: {filename}')

        # Lanzar trajectory_publisher con el archivo elegido
        p = subprocess.Popen([
            'ros2', 'run', 'vehicle_controller', 'trajectory_publisher',
            '--ros-args', '-p', f'trajectory_file:={filename}'
        ])
        self._processes.append(p)

        # Esperar a que TOPP reciba la trayectoria y se la envíe al MPC
        self._traj_timer = self.create_timer(2.0, self._on_traj_timer)

    def _on_traj_timer(self):
        self._traj_timer.cancel()
        self._traj_timer = None

        msg = Bool()
        msg.data = True
        self.start_pub.publish(msg)
        self._episode_active = True
        self.get_logger().info('Episodio activo — /rl/start publicado')

    # ─────────────────────────────────────────────────────────────
    def reset_episode(self):
        """Llamar desde la lógica RL cuando el episodio termina."""
        self.get_logger().info('Reseteando episodio...')
        self._episode_active = False
        self._fused_ready    = False

        msg = Bool()
        msg.data = True
        self.reset_pub.publish(msg)

        # Tras el reset, esperar a odometry/fused de nuevo
        # La suscripción ya existe, solo reseteamos el flag
        self.get_logger().info('Esperando odometry/fused para nuevo episodio...')

    # ─────────────────────────────────────────────────────────────
    def _pick_random_trajectory(self):
        files = glob.glob(os.path.join(self.trajectories_dir, '*.csv'))
        if not files:
            return None
        return random.choice(files)

    # ─────────────────────────────────────────────────────────────
    def shutdown(self):
        for p in self._processes:
            p.terminate()


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