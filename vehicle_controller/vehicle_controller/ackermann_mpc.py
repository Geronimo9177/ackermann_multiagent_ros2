#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import casadi as ca

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped
from tf_transformations import euler_from_quaternion


class AckermannMPC(Node):

    def __init__(self):

        super().__init__('ackermann_mpc')

        # Parámetros MPC
        self.dt = 0.05
        self.N = 20
        self.L = 0.335

        self.max_steer = 0.5
        self.max_speed = 1.0
        self.max_accel = 1.5

        self.state = np.zeros(3)
        self.prev_u = np.zeros(2)
        self.trajectory = None

        self.last_solution = None
        
        self.odom_received = False
        self.path_received = False
        
        # Índice actual en la trayectoria
        self.current_idx = None
        
        # Ventana de búsqueda hacia adelante
        self.search_window = 10

        self.create_subscription(
            Odometry,
            '/ground_truth_odom',
            self.odom_cb,
            10)

        self.create_subscription(
            Path,
            '/trajectory',
            self.path_cb,
            10)

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10)

        self.get_logger().info('Setting up MPC solver...')
        self.setup_mpc()
        self.get_logger().info('MPC solver ready!')

        self.create_timer(self.dt, self.control_loop)
        self.create_timer(2.0, self.debug_status)

    def debug_status(self):
        self.get_logger().info(
            f'Odom: {self.odom_received} | '
            f'Path: {self.path_received} | '
            f'Idx: {self.current_idx} | '
            f'State: [{self.state[0]:.2f}, {self.state[1]:.2f}, {np.degrees(self.state[2]):.1f}°]'
        )

    def setup_mpc(self):

        nx, nu = 3, 2

        X = ca.MX.sym('X', nx, self.N+1)
        U = ca.MX.sym('U', nu, self.N)
        P = ca.MX.sym('P', nx + 3*self.N)

        Q  = np.diag([50,  50,  8])   # ✅ Más tracking
        R  = np.diag([0.05, 0.3])     # ✅ Menos penalización de velocidad
        Rd = np.diag([1.0,  2.0])     # ✅ Menos suavizado

        cost = 0
        g    = []

        for k in range(self.N):

            st  = X[:, k]
            con = U[:, k]
            ref = P[nx + 3*k : nx + 3*k + 3]

            err = st - ref
            err_norm = ca.vertcat(
                err[0],
                err[1],
                ca.atan2(ca.sin(err[2]), ca.cos(err[2]))
            )

            cost += ca.mtimes([err_norm.T, Q, err_norm])
            cost += ca.mtimes([con.T, R, con])

            if k > 0:
                du = U[:, k] - U[:, k-1]
                cost += ca.mtimes([du.T, Rd, du])

            x_next = ca.vertcat(
                st[0] + con[0] * ca.cos(st[2]) * self.dt,
                st[1] + con[0] * ca.sin(st[2]) * self.dt,
                st[2] + (con[0] / self.L) * ca.tan(con[1]) * self.dt
            )
            g.append(X[:, k+1] - x_next)

        g.append(X[:, 0] - P[0:nx])

        OPT = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 150,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_obj_change_tol': 1e-4
        }

        self.solver = ca.nlpsol(
            'solver', 'ipopt',
            dict(f=cost, x=OPT, g=ca.vertcat(*g), p=P),
            opts
        )

        self.lbx, self.ubx = [], []

        for _ in range(self.N+1):
            self.lbx += [-ca.inf] * 3
            self.ubx += [ ca.inf] * 3

        for _ in range(self.N):
            self.lbx += [-self.max_speed, -self.max_steer]
            self.ubx += [ self.max_speed,  self.max_steer]

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.state = np.array([p.x, p.y, yaw])
        if not self.odom_received:
            self.get_logger().info('First odometry received!')
            self.odom_received = True

    def path_cb(self, msg):
        self.trajectory  = msg.poses
        self.current_idx = None   # Reset al recibir nueva trayectoria
        if not self.path_received:
            first = msg.poses[0].pose.position
            self.get_logger().info(
                f'Trajectory received: {len(msg.poses)} pts | '
                f'First: ({first.x:.2f}, {first.y:.2f})'
            )
            self.path_received = True

    def normalize_angle(self, angle):
        while angle >  np.pi: angle -= 2*np.pi
        while angle < -np.pi: angle += 2*np.pi
        return angle

    # ==================================================
    # Búsqueda inicial: recorre TODA la trayectoria
    # ==================================================
    def find_closest_point_index(self):
        best_idx  = 0
        best_cost = float('inf')

        for i, pose in enumerate(self.trajectory):
            pos  = pose.pose.position
            dist = np.hypot(pos.x - self.state[0],
                            pos.y - self.state[1])

            angle_to = np.arctan2(pos.y - self.state[1],
                                  pos.x - self.state[0])
            head_err = abs(self.normalize_angle(angle_to - self.state[2]))

            cost = dist + (5.0 * head_err if head_err < np.pi/2 else 20.0 * head_err)

            if cost < best_cost:
                best_cost = cost
                best_idx  = i

        self.get_logger().info(f'Closest point found: idx={best_idx}')
        return best_idx

    # ==================================================
    # Actualización: busca SOLO HACIA ADELANTE
    # ==================================================
    def update_current_index(self):

        # Primera vez: buscar punto inicial
        if self.current_idx is None:
            self.current_idx = self.find_closest_point_index()
            return

        n = len(self.trajectory)

        # Buscar el punto más cercano en los próximos `search_window` puntos
        best_idx  = self.current_idx
        best_dist = float('inf')

        for i in range(self.search_window + 1):
            idx  = (self.current_idx + i) % n
            pos  = self.trajectory[idx].pose.position
            dist = np.hypot(pos.x - self.state[0],
                            pos.y - self.state[1])
            if dist < best_dist:
                best_dist = dist
                best_idx  = idx

        # Solo avanzar, nunca retroceder
        if best_idx != self.current_idx:
            self.get_logger().info(
                f'Index advanced: {self.current_idx} → {best_idx} '
                f'(dist={best_dist:.2f}m)',
                throttle_duration_sec=1.0
            )
            self.current_idx = best_idx

    # ==================================================
    # CONTROL LOOP
    # ==================================================
    def control_loop(self):

        if self.trajectory is None or not self.odom_received:
            return

        if len(self.trajectory) < self.N:
            self.get_logger().warn('Trajectory too short!', throttle_duration_sec=2.0)
            self.publish_cmd(np.zeros(2))
            return

        try:
            self.update_current_index()

            n   = len(self.trajectory)
            ref = []

            for i in range(self.N):
                idx  = (self.current_idx + i) % n
                pos  = self.trajectory[idx].pose.position
                quat = self.trajectory[idx].pose.orientation
                _, _, yaw = euler_from_quaternion(
                    [quat.x, quat.y, quat.z, quat.w])
                ref += [pos.x, pos.y, yaw]

            P  = np.concatenate([self.state, ref])
            x0 = self.last_solution if self.last_solution is not None \
                 else np.zeros(3*(self.N+1) + 2*self.N)

            sol = self.solver(
                x0=x0, p=P,
                lbg=0, ubg=0,
                lbx=self.lbx, ubx=self.ubx
            )

            stats = self.solver.stats()
            if not stats['success']:
                self.get_logger().warn(
                    f'Solver: {stats["return_status"]}',
                    throttle_duration_sec=1.0
                )
                self.publish_cmd(self.prev_u * 0.5)
                return

            opt = sol['x'].full().flatten()
            self.last_solution = opt

            nx = 3*(self.N+1)
            u  = opt[nx:nx+2]
            self.prev_u = u

            ref_pt = self.trajectory[self.current_idx].pose.position
            error  = np.hypot(self.state[0] - ref_pt.x,
                              self.state[1] - ref_pt.y)

            self.get_logger().info(
                f'Idx: {self.current_idx}/{n} | '
                f'Ref: ({ref_pt.x:.1f}, {ref_pt.y:.1f}) | '
                f'Error: {error:.3f}m | '
                f'v: {u[0]:.2f}m/s | δ: {np.degrees(u[1]):.1f}°',
                throttle_duration_sec=0.5
            )

            self.publish_cmd(u)

        except Exception as e:
            self.get_logger().error(f'MPC failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.publish_cmd(np.zeros(2))

    def publish_cmd(self, u):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        v     = float(u[0])
        delta = float(u[1])
        msg.twist.linear.x  = v
        # ackermann_steering_controller expects steering angle in angular.z
        #msg.twist.angular.z = delta
        msg.twist.angular.z = v / self.L * np.tan(delta)
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = AckermannMPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()