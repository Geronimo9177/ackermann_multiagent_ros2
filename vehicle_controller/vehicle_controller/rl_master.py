#!/usr/bin/env python3
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


class RLMaster(Node):

    # ── Condiciones de fin de episodio ──────────────────────────
    CRASH_SPEED_THRESHOLD  = 0.05   # m/s  — velocidad "casi cero" = choque
    CRASH_SPEED_DURATION   = 1.0    # s    — cuánto tiempo debe mantenerse
    FALL_Z_THRESHOLD       = -1.0   # m    — Z por debajo de esto = caída del mapa
    MIN_INITIAL_SPEED      = 0.3    # m/s  — el vehículo debe superar esta vel

    RESULT_RUNNING  = 0
    RESULT_SUCCESS  = 1
    RESULT_CRASH    = 2
    RESULT_FALL     = 3

    def __init__(self):
        super().__init__('rl_master')

        self.declare_parameter('training_mode', True)
        self.training_mode = self.get_parameter('training_mode').value

        # Solo gestionamos el proceso de la trayectoria
        self._traj_process   = None 

        self._fused_ready    = False
        self._episode_active = False
        self._start_timer    = None
        self._traj_timer     = None

        # Estado para detección de fin de episodio
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

        # Subscribers
        self.topp_sub = self.create_subscription(
            Float64MultiArray, '/trajectory_topp', self._topp_cb, 1)
        
        self.fused_sub = self.create_subscription(
            Odometry, '/odometry/fused', self._fused_cb, 10)

        self.success_sub = self.create_subscription(
            Bool, '/mpc/success', self._success_cb, 1)

        self.gt_sub = self.create_subscription(
            Odometry, '/ground_truth_odom', self._gt_cb, 10)
        
        self.get_logger().info('RLMaster iniciado — esperando /odometry/fused...')

    # ── Callbacks ────────────────────────────────────────────────
    def _success_cb(self, _msg: Bool):
        """El MPC llegó al goal — episodio exitoso."""
        if self._episode_active:
            self.get_logger().info('SUCCESS — vehículo llegó al destino')
            self._end_episode(self.RESULT_SUCCESS)

    def _gt_cb(self, msg: Odometry):
        """Velocidad ground truth para detección de crash."""
        if not self._episode_active:
            return
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.sqrt(vx**2 + vy**2)

        z_pos = msg.pose.pose.position.z

        # Caída del mapa
        if z_pos < self.FALL_Z_THRESHOLD:
            self.get_logger().warn(f'CAÍDA detectada (z={z_pos:.2f}m)')
            self._end_episode(self.RESULT_FALL)
            return
        
        now = self.get_clock().now().nanoseconds * 1e-9

        if speed > self.MIN_INITIAL_SPEED:
            self._vehicle_moved = True

        if self._vehicle_moved:
            if speed < self.CRASH_SPEED_THRESHOLD:
                if self._low_speed_since is None:
                    self._low_speed_since = now
                elif now - self._low_speed_since >= self.CRASH_SPEED_DURATION:
                    self.get_logger().warn(f'CHOQUE detectado (v={speed:.3f} m/s)')
                    self._end_episode(self.RESULT_CRASH)
            else:
                self._low_speed_since = None

    def _fused_cb(self, msg: Odometry):
        if not self._fused_ready:
            self._fused_ready = True
            self.get_logger().info('odometry/fused activo — arrancando en 2s...')
            self._start_timer = self.create_timer(2.0, self._on_start_timer)
            return

    def _topp_cb(self, _msg):
        if self._episode_active or not self._episode_count:
            return 
        if self._topp_ready:
            return
        
        self._topp_ready = True
        self.get_logger().info('TOPP listo — arrancando episodio en 0.5s...')
        self._traj_timer = self.create_timer(0.5, self._on_traj_timer)

    # ── Timers de arranque ───────────────────────────────────────
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
        self.get_logger().info(f'[Ep {self._episode_count}] Activo — /rl/start publicado')

    # ── Ciclo de episodio ────────────────────────────────────────
    def start_episode(self):
        traj_file = self._pick_random_trajectory()
        if traj_file is None:
            self.get_logger().error(f'No hay trayectorias en: {self.trajectories_dir}')
            return

        self._episode_count += 1
        self._low_speed_since = None
        self._vehicle_moved   = False
        self._topp_ready      = False

        filename = os.path.basename(traj_file)
        self.get_logger().info(f'[Ep {self._episode_count}] Trayectoria: {filename}')

        # Aniquilación inmediata del proceso anterior para evitar congelamientos
        if self._traj_process is not None:
            try:
                self._traj_process.kill()
            except Exception:
                pass
            self._traj_process = None

        self._traj_process = subprocess.Popen([
            'ros2', 'run', 'vehicle_controller', 'trajectory_publisher',
            '--ros-args', '-p', f'trajectory_file:={filename}'
        ])
    
    def _end_episode(self, result: int):
        """Punto único de salida de episodio."""
        self._episode_active = False

        result_names = {
            self.RESULT_SUCCESS: 'SUCCESS',
            self.RESULT_CRASH:   'CRASH',
            self.RESULT_FALL:    'FALL',
        }
        self.get_logger().info(
            f'[Ep {self._episode_count}] FIN → {result_names.get(result, "?")}')

        # Publicar resultado para el agente RL
        msg_r = Int32()
        msg_r.data = result
        self.result_pub.publish(msg_r)

        # Señal de reset a Webots
        msg_b = Bool()
        msg_b.data = True
        self.reset_pub.publish(msg_b)

        self._fused_ready = False
        self.get_logger().info('Esperando /odometry/fused para nuevo episodio...')

    # ── Helpers ──────────────────────────────────────────────────
    def _pick_random_trajectory(self):
        files = glob.glob(os.path.join(self.trajectories_dir, '*.csv'))
        return random.choice(files) if files else None

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