#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import math

class OdometryComparer(Node):
    def __init__(self):
        super().__init__('odom_comparer')
        
        # Parámetros
        self.declare_parameter('window_size', 1000)
        self.declare_parameter('update_rate', 100.0)
        
        self.window_size = self.get_parameter('window_size').value
        update_rate = self.get_parameter('update_rate').value
        
        # Subscribers
        self.gt_sub = self.create_subscription(
            Odometry,
            '/ground_truth_odom',
            self.gt_callback,
            10)
        
        self.ctrl_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.ctrl_callback,
            10)
        
        # Datos almacenados
        self.gt_data = {
            'x': deque(maxlen=self.window_size),
            'y': deque(maxlen=self.window_size),
            'yaw': deque(maxlen=self.window_size),
            'vx': deque(maxlen=self.window_size),
            'vyaw': deque(maxlen=self.window_size),
            'time': deque(maxlen=self.window_size)
        }
        
        self.ctrl_data = {
            'x': deque(maxlen=self.window_size),
            'y': deque(maxlen=self.window_size),
            'yaw': deque(maxlen=self.window_size),
            'vx': deque(maxlen=self.window_size),
            'vyaw': deque(maxlen=self.window_size),
            'time': deque(maxlen=self.window_size)
        }
        
        # Errores
        self.errors = {
            'position': deque(maxlen=self.window_size),
            'x': deque(maxlen=self.window_size),
            'y': deque(maxlen=self.window_size),
            'yaw': deque(maxlen=self.window_size),
            'vx': deque(maxlen=self.window_size),
            'vyaw': deque(maxlen=self.window_size),
            'time': deque(maxlen=self.window_size)
        }
        
        self.start_time = self.get_clock().now()
        
        # Estado actual
        self.gt_pose = None
        self.ctrl_pose = None
        
        # Timer para calcular métricas
        self.metrics_timer = self.create_timer(1.0 / update_rate, self.calculate_metrics)
        
        # Timer para imprimir estadísticas
        self.stats_timer = self.create_timer(5.0, self.print_statistics)
        
        # Configurar matplotlib
        plt.ion()
        self.setup_plots()
        
        # Timer para actualizar gráficas
        self.plot_timer = self.create_timer(1.0 / update_rate, self.update_plots)
        
        self.get_logger().info('Odometry Comparer Started')
        self.get_logger().info('Comparing /ground_truth_odom vs /ackermann_steering_controller/odometry')
    
    def gt_callback(self, msg):
        self.gt_pose = msg
    
    def ctrl_callback(self, msg):
        self.ctrl_pose = msg
    
    def quaternion_to_yaw(self, quat):
        """Convierte quaternion a yaw"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def normalize_angle(self, angle):
        """Normaliza ángulo a [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def calculate_metrics(self):
        """Calcula métricas de error"""
        if self.gt_pose is None or self.ctrl_pose is None:
            return
        
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        # Extraer posiciones
        gt_x = self.gt_pose.pose.pose.position.x
        gt_y = self.gt_pose.pose.pose.position.y
        gt_yaw = self.quaternion_to_yaw(self.gt_pose.pose.pose.orientation)
        gt_vx = self.gt_pose.twist.twist.linear.x
        gt_vyaw = self.gt_pose.twist.twist.angular.z
        
        ctrl_x = self.ctrl_pose.pose.pose.position.x
        ctrl_y = self.ctrl_pose.pose.pose.position.y
        ctrl_yaw = self.quaternion_to_yaw(self.ctrl_pose.pose.pose.orientation)
        ctrl_vx = self.ctrl_pose.twist.twist.linear.x
        ctrl_vyaw = self.ctrl_pose.twist.twist.angular.z
        
        # Guardar datos
        self.gt_data['x'].append(gt_x)
        self.gt_data['y'].append(gt_y)
        self.gt_data['yaw'].append(gt_yaw)
        self.gt_data['vx'].append(gt_vx)
        self.gt_data['vyaw'].append(gt_vyaw)
        self.gt_data['time'].append(current_time)
        
        self.ctrl_data['x'].append(ctrl_x)
        self.ctrl_data['y'].append(ctrl_y)
        self.ctrl_data['yaw'].append(ctrl_yaw)
        self.ctrl_data['vx'].append(ctrl_vx)
        self.ctrl_data['vyaw'].append(ctrl_vyaw)
        self.ctrl_data['time'].append(current_time)
        
        # Calcular errores
        error_x = ctrl_x - gt_x
        error_y = ctrl_y - gt_y
        error_position = math.sqrt(error_x**2 + error_y**2)
        error_yaw = self.normalize_angle(ctrl_yaw - gt_yaw)
        error_vx = ctrl_vx - gt_vx
        error_vyaw = ctrl_vyaw - gt_vyaw
        
        self.errors['x'].append(error_x)
        self.errors['y'].append(error_y)
        self.errors['position'].append(error_position)
        self.errors['yaw'].append(error_yaw)
        self.errors['vx'].append(error_vx)
        self.errors['vyaw'].append(error_vyaw)
        self.errors['time'].append(current_time)
    
    def setup_plots(self):
        """Configura las gráficas"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Odometry Comparison: Ground Truth vs Controller', 
                         fontsize=16, fontweight='bold')
        
        # 1. Trayectoria 2D
        self.ax1 = plt.subplot(3, 3, 1)
        self.ax1.set_title('2D Trajectory')
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')
        self.line_gt_traj, = self.ax1.plot([], [], 'g-', label='Ground Truth', linewidth=2)
        self.line_ctrl_traj, = self.ax1.plot([], [], 'b--', label='Controller', linewidth=2)
        self.ax1.legend()
        
        # 2. Error de posición
        self.ax2 = plt.subplot(3, 3, 2)
        self.ax2.set_title('Position Error (2D Distance)')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Error (m)')
        self.ax2.grid(True, alpha=0.3)
        self.line_pos_error, = self.ax2.plot([], [], 'r-', linewidth=2)
        
        # 3. Error de yaw
        self.ax3 = plt.subplot(3, 3, 3)
        self.ax3.set_title('Yaw Error')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Error (deg)')
        self.ax3.grid(True, alpha=0.3)
        self.line_yaw_error, = self.ax3.plot([], [], 'm-', linewidth=2)
        
        # 4. Error X
        self.ax4 = plt.subplot(3, 3, 4)
        self.ax4.set_title('X Error')
        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('Error (m)')
        self.ax4.grid(True, alpha=0.3)
        self.line_x_error, = self.ax4.plot([], [], 'c-', linewidth=2)
        self.ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 5. Error Y
        self.ax5 = plt.subplot(3, 3, 5)
        self.ax5.set_title('Y Error')
        self.ax5.set_xlabel('Time (s)')
        self.ax5.set_ylabel('Error (m)')
        self.ax5.grid(True, alpha=0.3)
        self.line_y_error, = self.ax5.plot([], [], 'orange', linewidth=2)
        self.ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 6. Comparación de velocidad lineal
        self.ax6 = plt.subplot(3, 3, 6)
        self.ax6.set_title('Linear Velocity Comparison')
        self.ax6.set_xlabel('Time (s)')
        self.ax6.set_ylabel('Velocity (m/s)')
        self.ax6.grid(True, alpha=0.3)
        self.line_gt_vx, = self.ax6.plot([], [], 'g-', label='Ground Truth', linewidth=2)
        self.line_ctrl_vx, = self.ax6.plot([], [], 'b--', label='Controller', linewidth=2)
        self.ax6.legend()
        
        # 7. Comparación de velocidad angular
        self.ax7 = plt.subplot(3, 3, 7)
        self.ax7.set_title('Angular Velocity Comparison')
        self.ax7.set_xlabel('Time (s)')
        self.ax7.set_ylabel('Velocity (deg/s)')
        self.ax7.grid(True, alpha=0.3)
        self.line_gt_vyaw, = self.ax7.plot([], [], 'g-', label='Ground Truth', linewidth=2)
        self.line_ctrl_vyaw, = self.ax7.plot([], [], 'b--', label='Controller', linewidth=2)
        self.ax7.legend()
        
        # 8. Error de velocidad angular
        self.ax8 = plt.subplot(3, 3, 8)
        self.ax8.set_title('Angular Velocity Error')
        self.ax8.set_xlabel('Time (s)')
        self.ax8.set_ylabel('Error (deg/s)')
        self.ax8.grid(True, alpha=0.3)
        self.line_vyaw_error, = self.ax8.plot([], [], 'purple', linewidth=2)
        self.ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 9. Estadísticas
        self.ax9 = plt.subplot(3, 3, 9)
        self.ax9.axis('off')
        self.stats_text = self.ax9.text(0.05, 0.5, '', fontsize=10, family='monospace',
                                       verticalalignment='center')
        
        plt.tight_layout()
    
    def update_plots(self):
        """Actualiza las gráficas en tiempo real"""
        if len(self.gt_data['x']) < 2:
            return
        
        # Trayectoria
        self.line_gt_traj.set_data(list(self.gt_data['x']), list(self.gt_data['y']))
        self.line_ctrl_traj.set_data(list(self.ctrl_data['x']), list(self.ctrl_data['y']))
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Error de posición
        self.line_pos_error.set_data(list(self.errors['time']), 
                                     list(self.errors['position']))
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Error de yaw
        yaw_errors_deg = [math.degrees(y) for y in self.errors['yaw']]
        self.line_yaw_error.set_data(list(self.errors['time']), yaw_errors_deg)
        self.ax3.relim()
        self.ax3.autoscale_view()
        
        # Error X
        self.line_x_error.set_data(list(self.errors['time']), list(self.errors['x']))
        self.ax4.relim()
        self.ax4.autoscale_view()
        
        # Error Y
        self.line_y_error.set_data(list(self.errors['time']), list(self.errors['y']))
        self.ax5.relim()
        self.ax5.autoscale_view()
        
        # Velocidad lineal
        self.line_gt_vx.set_data(list(self.gt_data['time']), list(self.gt_data['vx']))
        self.line_ctrl_vx.set_data(list(self.ctrl_data['time']), list(self.ctrl_data['vx']))
        self.ax6.relim()
        self.ax6.autoscale_view()
        
        # Velocidad angular
        gt_vyaw_deg = [math.degrees(v) for v in self.gt_data['vyaw']]
        ctrl_vyaw_deg = [math.degrees(v) for v in self.ctrl_data['vyaw']]
        self.line_gt_vyaw.set_data(list(self.gt_data['time']), gt_vyaw_deg)
        self.line_ctrl_vyaw.set_data(list(self.ctrl_data['time']), ctrl_vyaw_deg)
        self.ax7.relim()
        self.ax7.autoscale_view()
        
        # Error velocidad angular
        vyaw_errors_deg = [math.degrees(v) for v in self.errors['vyaw']]
        self.line_vyaw_error.set_data(list(self.errors['time']), vyaw_errors_deg)
        self.ax8.relim()
        self.ax8.autoscale_view()
        
        # Actualizar estadísticas
        self.update_statistics_text()
        
        # Refrescar
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update_statistics_text(self):
        """Actualiza texto de estadísticas"""
        if len(self.errors['position']) == 0:
            return
        
        pos_errors = list(self.errors['position'])
        x_errors = list(self.errors['x'])
        y_errors = list(self.errors['y'])
        yaw_errors = [math.degrees(y) for y in self.errors['yaw']]
        vyaw_errors = [math.degrees(v) for v in self.errors['vyaw']]
        
        stats = f"""
╔═══════════════════════════════╗
║   ODOMETRY ERROR STATS        ║
╠═══════════════════════════════╣
║ Position Error (2D):          ║
║   Mean:  {np.mean(pos_errors):7.4f} m      ║
║   Std:   {np.std(pos_errors):7.4f} m      ║
║   Max:   {np.max(pos_errors):7.4f} m      ║
║   RMSE:  {np.sqrt(np.mean(np.square(pos_errors))):7.4f} m      ║
║                               ║
║ X Error:                      ║
║   Mean:  {np.mean(x_errors):7.4f} m      ║
║   Std:   {np.std(x_errors):7.4f} m      ║
║                               ║
║ Y Error:                      ║
║   Mean:  {np.mean(y_errors):7.4f} m      ║
║   Std:   {np.std(y_errors):7.4f} m      ║
║                               ║
║ Yaw Error:                    ║
║   Mean:  {np.mean(yaw_errors):7.2f}°       ║
║   Std:   {np.std(yaw_errors):7.2f}°       ║
║   Max:   {np.max(np.abs(yaw_errors)):7.2f}°       ║
║   RMSE:  {np.sqrt(np.mean(np.square(yaw_errors))):7.2f}°       ║
║                               ║
║ Angular Vel Error:            ║
║   Mean:  {np.mean(vyaw_errors):7.2f}°/s    ║
║   Std:   {np.std(vyaw_errors):7.2f}°/s    ║
║                               ║
║ Samples: {len(pos_errors):6d}             ║
╚═══════════════════════════════╝
        """
        
        self.stats_text.set_text(stats)
    
    def print_statistics(self):
        """Imprime estadísticas en consola"""
        if len(self.errors['position']) == 0:
            self.get_logger().warn('No data available yet')
            return
        
        pos_errors = list(self.errors['position'])
        yaw_errors = [math.degrees(y) for y in self.errors['yaw']]
        
        rmse_pos = np.sqrt(np.mean(np.square(pos_errors)))
        rmse_yaw = np.sqrt(np.mean(np.square(yaw_errors)))
        
        self.get_logger().info(
            f'\n{"="*60}\n'
            f'Position RMSE: {rmse_pos:.4f} m | '
            f'Mean: {np.mean(pos_errors):.4f} m | '
            f'Max: {np.max(pos_errors):.4f} m\n'
            f'Yaw RMSE: {rmse_yaw:.2f}° | '
            f'Mean: {np.mean(yaw_errors):.2f}° | '
            f'Max: {np.max(np.abs(yaw_errors)):.2f}°\n'
            f'Samples: {len(pos_errors)}\n'
            f'{"="*60}'
        )


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = OdometryComparer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        rclpy.shutdown()


if __name__ == '__main__':
    main()