#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import casadi as ca
import time

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Float64MultiArray
from tf_transformations import euler_from_quaternion


# Orthogonal projection onto a line segment
def get_projection(p, a, b):
    v    = b - a
    w    = p - a
    v_sq = np.dot(v, v)
    if v_sq == 0.0:
        return np.linalg.norm(p - a), 0.0
    t         = np.dot(w, v) / v_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    dist      = np.linalg.norm(p - (a + t_clamped * v))
    return dist, t_clamped


class AckermannMPC(Node):

    def __init__(self):
        super().__init__('ackermann_mpc')

        # MPC parameters
        self.dt        = 0.05
        self.N         = 20
        self.L         = 0.335
        self.max_steer = 0.785398163  # 45 degrees in radians
        self.max_speed = 1.0

        # Internal state
        self.state         = np.zeros(3)
        self.prev_u        = np.zeros(2)
        self.last_solution = None

        # Arrays from the MultiArray message (TOPP node)
        self.xy_arr  = None   # (N, 2)
        self.yaw_arr = None   # (N,)
        self.v_arr   = None   # (N,)
        self.s_arr   = None   # (N,)

        self.odom_received = False
        self.path_received = False

        # Path tracking
        self.current_idx   = None
        self.current_t     = 0.0
        self.search_window = 10
        self.s_search      = 3

        # Goal completion criteria
        self.goal_tolerance =  0.10
        self.route_completed = False

        # Subscribers
        self.create_subscription(Odometry, '/ground_truth_odom', self.odom_cb, 10)
        self.create_subscription(Float64MultiArray, '/trajectory_topp',
                                 self.trajectory_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Float64MultiArray, '/mpc/debug', 10)
        self.predicted_path_pub = self.create_publisher(Path, '/mpc/predicted_path', 10)

        self.get_logger().info('Setting up MPC solver...')
        self.setup_mpc()
        self.get_logger().info('MPC solver ready!')

        self.create_timer(self.dt, self.control_loop)
        self.create_timer(2.0,     self.debug_status)

    def debug_status(self):
        self.get_logger().info(
            f'Odom: {self.odom_received} | Path: {self.path_received} | '
            f'Idx: {self.current_idx} | '
            f'State: [{self.state[0]:.2f}, {self.state[1]:.2f}, '
            f'{np.degrees(self.state[2]):.1f}°]'
        )

    # Setup MPC: cost function in Frenet coordinates (s, l, φ, v)
    def setup_mpc(self):
        nx, nu = 3, 2
        # Reference per step: [x_ref, y_ref, yaw_ref, v_ref]
        # yaw_ref == ψ_ref (segment angle used for Frenet frame rotation)
        n_ref  = 4

        X = ca.MX.sym('X', nx, self.N + 1)
        U = ca.MX.sym('U', nu, self.N)
        P = ca.MX.sym('P', nx + n_ref * self.N)

        # Cost weights (from AutoMPC paper equation 6)
        Q_lon = 5.0    # q1: longitudinal weight
        Q_lat = 80.0   # q2: lateral weight (>> Q_lon for path following)
        Q_yaw = 8.0    # q3: yaw weight
        Q_v   = 5.0    # q4: velocity weight

        R  = np.diag([0.05, 0.3])   # Actuator penalty (v_cmd, δ)
        Rd = np.diag([1.0,  2.0])   # Rate-of-change smoothing

        cost = 0
        g    = []

        for k in range(self.N):
            st  = X[:, k]           # [x, y, φ]
            con = U[:, k]           # [v_cmd, δ]
            ref = P[nx + n_ref * k : nx + n_ref * k + n_ref]
            # ref[0]=x_ref, ref[1]=y_ref, ref[2]=ψ_ref, ref[3]=v_ref

            # Rotate position error into the segment's local frame
            # Equivalent to projecting onto tangent (s) and normal (l) directions
            dx = st[0] - ref[0]
            dy = st[1] - ref[1]
            psi = ref[2]   # Segment yaw angle

            # Longitudinal error: projection along the segment direction
            e_s =  ca.cos(psi) * dx + ca.sin(psi) * dy

            # Lateral error: projection perpendicular to the segment
            # Positive = left of the segment
            e_l = -ca.sin(psi) * dx + ca.cos(psi) * dy

            # Yaw error (normalized)
            e_yaw = ca.atan2(ca.sin(st[2] - ref[2]), ca.cos(st[2] - ref[2]))

            # Velocity error (commanded vs TOPP reference)
            e_v = con[0] - ref[3]

            # Cost function (AutoMPC equation 6)
            cost += Q_lon * e_s**2
            cost += Q_lat * e_l**2
            cost += Q_yaw * e_yaw**2
            cost += Q_v   * e_v**2
            cost += ca.mtimes([con.T, R, con])

            if k > 0:
                du    = U[:, k] - U[:, k - 1]
                cost += ca.mtimes([du.T, Rd, du])

            # Bicycle kinematics dynamics
            x_next = ca.vertcat(
                st[0] + con[0] * ca.cos(st[2]) * self.dt,
                st[1] + con[0] * ca.sin(st[2]) * self.dt,
                st[2] + (con[0] / self.L) * ca.tan(con[1]) * self.dt
            )
            g.append(X[:, k + 1] - x_next)

        # Initial condition constraint
        g.append(X[:, 0] - P[0:nx])

        OPT  = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        opts = {
            'ipopt.print_level':               0,
            'print_time':                      0,
            'ipopt.max_iter':                150,
            'ipopt.acceptable_tol':          1e-4,
            'ipopt.acceptable_obj_change_tol': 1e-4,
        }
        self.solver = ca.nlpsol('solver', 'ipopt',
                                dict(f=cost, x=OPT, g=ca.vertcat(*g), p=P),
                                opts)

        self.lbx, self.ubx = [], []
        for _ in range(self.N + 1):
            self.lbx += [-ca.inf] * 3
            self.ubx += [ ca.inf] * 3
        for _ in range(self.N):
            self.lbx += [-self.max_speed, -self.max_steer]
            self.ubx += [ self.max_speed,  self.max_steer]

        self.n_ref = n_ref

    # Subscription callbacks
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.state = np.array([p.x, p.y, yaw])
        if not self.odom_received:
            self.get_logger().info('First odometry received!')
            self.odom_received = True

    def trajectory_cb(self, msg: Float64MultiArray):
        """Deserialize Float64MultiArray to numpy arrays.
        Layout: [x, y, yaw, v, s] per point.
        """
        n      = msg.layout.dim[0].size
        fields = msg.layout.dim[1].size

        data = np.array(msg.data).reshape(n, fields)
        self.xy_arr  = data[:, 0:2]
        self.yaw_arr = data[:, 2]
        self.v_arr   = data[:, 3]
        self.s_arr   = data[:, 4]

        self.current_idx = None
        self.current_t   = 0.0
        self.route_completed = False
        self.last_solution = None

        if not self.path_received:
            self.get_logger().info(
                f'Trajectory TOPP: {n} pts | '
                f'v [{self.v_arr.min():.2f}, {self.v_arr.max():.2f}] m/s'
            )
            self.path_received = True

    # Geometry helpers
    def get_segment(self, i, n):
        idx_a = min(i, n - 2)
        return self.xy_arr[idx_a], self.xy_arr[idx_a + 1]

    def get_v_ref_at(self, idx, t):
        idx_a = min(idx, len(self.v_arr) - 2)
        return float((1.0 - t) * self.v_arr[idx_a] + t * self.v_arr[idx_a + 1])
    
    def get_yaw_ref_at(self, idx, t):
        """Circular interpolation of reference yaw angle."""
        idx_a = min(idx, len(self.yaw_arr) - 2)
        a0, a1 = self.yaw_arr[idx_a], self.yaw_arr[idx_a + 1]
        return float(a0 + t * np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)))

    # Index update (look-ahead search)
    def update_current_index(self):
        if self.current_idx is None:
            self.current_idx = 0
            self.current_t   = 0.0
            return

        n = len(self.xy_arr)
        if self.current_idx >= n - 2:
            return

        p         = self.state[:2]
        best_idx  = self.current_idx
        best_dist = float('inf')
        best_t    = self.current_t

        start = max(0,     self.current_idx - self.s_search)
        end   = min(n - 1, self.current_idx + self.search_window)

        for idx in range(start, end):
            a, b = self.get_segment(idx, n)
            dist, t = get_projection(p, a, b)
            if dist < best_dist:
                best_dist = dist
                best_idx  = idx
                best_t    = t

        if best_t >= 0.9999 and best_idx < n - 2:
            best_idx += 1
            best_t    = 0.0

        if best_idx != self.current_idx:
            self.get_logger().info(
                f'Segment: {self.current_idx} → {best_idx} '
                f'(t={best_t:.2f}, dist={best_dist:.3f}m)',
                throttle_duration_sec=1.0
            )
            self.current_idx = best_idx
        self.current_t = best_t

    # Warm start for MPC solver
    def get_warm_start(self, opt):
        nx_states      = 3 * (self.N + 1)
        states         = opt[:nx_states].reshape(self.N + 1, 3)
        inputs         = opt[nx_states:].reshape(self.N, 2)
        states_shifted = np.roll(states, -1, axis=0); states_shifted[-1] = states[-1]
        inputs_shifted = np.roll(inputs, -1, axis=0); inputs_shifted[-1] = inputs[-1]
        return np.concatenate([states_shifted.flatten(), inputs_shifted.flatten()])

    def should_complete_route(self):
        if self.xy_arr is None or self.current_idx is None:
            return False
        goal = self.xy_arr[-1]
        goal_distance = float(np.hypot(self.state[0] - goal[0], self.state[1] - goal[1]))
        return goal_distance<= self.goal_tolerance

    def make_pose_stamped(self, x, y, yaw, stamp):
        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.header.stamp = stamp
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = float(np.sin(yaw * 0.5))
        pose.pose.orientation.w = float(np.cos(yaw * 0.5))
        return pose

    def publish_predicted_path(self, opt, stamp):
        nx_states = 3 * (self.N + 1)
        states = opt[:nx_states].reshape(self.N + 1, 3)

        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp = stamp
        msg.poses = [self.make_pose_stamped(x, y, yaw, stamp) for x, y, yaw in states]
        self.predicted_path_pub.publish(msg)

    def publish_debug_metrics(self, x_now, y_now, yaw_now, x_ref, y_ref, yaw_ref,
                              v_cmd, v_ref, steer, solve_ms):
        msg = Float64MultiArray()
        msg.data = [
            float(x_now),
            float(y_now),
            float(yaw_now),
            float(x_ref),
            float(y_ref),
            float(yaw_ref),
            float(v_cmd),
            float(v_ref),
            float(steer),
            float(self.current_idx if self.current_idx is not None else -1),
            float(solve_ms),
        ]
        self.debug_pub.publish(msg)

    # Main control loop
    def control_loop(self):
        if self.xy_arr is None or not self.odom_received or self.route_completed:
            return


        n = len(self.xy_arr)
        if n < 2:
            self.get_logger().warn('Trajectory too short!', throttle_duration_sec=2.0)
            self.publish_cmd(np.zeros(2))
            return

        try:
            self.update_current_index()

            if self.current_idx >= int(0.95*(n-1)):
                if self.should_complete_route():
                    self.route_completed = True
                    self.path_received = False
                    self.current_idx   = None
                    self.publish_cmd(np.zeros(2))
                    self.get_logger().info(
                        f'Route completed (tol={self.goal_tolerance:.3f} m)'
                    )
                    return

            ref      = []
            path_idx = self.current_idx
            path_t   = self.current_t

            a, b    = self.get_segment(path_idx, n)
            seg_len = np.linalg.norm(b - a)

            for k in range(self.N):
                if seg_len > 0.001:
                    ref_pt  = a + path_t * (b - a)
                    ref_yaw = self.get_yaw_ref_at(path_idx, path_t)
                else:
                    ref_pt  = b
                    ref_yaw = self.state[2]

                v_ref_k = self.get_v_ref_at(path_idx, path_t)

                # ref = [x_ref, y_ref, ψ_ref, v_ref]
                # ψ_ref is the segment yaw used for Frenet frame rotation
                ref.extend([ref_pt[0], ref_pt[1], ref_yaw, v_ref_k])

                # Spatial advance along the path using v_ref_k
                dist_step = v_ref_k * self.dt
                while dist_step > 0 and path_idx < n - 2:
                    dist_remaining = (1.0 - path_t) * seg_len
                    if dist_step <= dist_remaining:
                        path_t   += dist_step / seg_len
                        dist_step = 0
                    else:
                        dist_step -= dist_remaining
                        path_idx  += 1
                        path_t     = 0.0
                        a, b       = self.get_segment(path_idx, n)
                        seg_len    = np.linalg.norm(b - a)

                if path_idx >= n - 2 and dist_step > 0:
                    path_t = 1.0

            P_vec = np.concatenate([self.state, ref])
            x0    = (self.get_warm_start(self.last_solution)
                     if self.last_solution is not None
                     else np.zeros(3 * (self.N + 1) + 2 * self.N))

            t0 = time.perf_counter()
            sol = self.solver(x0=x0, p=P_vec, lbg=0, ubg=0,
                              lbx=self.lbx, ubx=self.ubx)
            solve_ms = 1000.0 * (time.perf_counter() - t0)

            if not self.solver.stats()['success']:
                self.get_logger().warn(
                    f'Solver: {self.solver.stats()["return_status"]}',
                    throttle_duration_sec=1.0)
                self.publish_cmd(self.prev_u * 0.5)
                return

            opt            = sol['x'].full().flatten()
            self.last_solution = opt
            u              = opt[3 * (self.N + 1) : 3 * (self.N + 1) + 2]
            self.prev_u    = u

            # Compute Frenet frame errors for logging
            a_act, b_act = self.get_segment(self.current_idx, n)
            ref_pt_now   = a_act + self.current_t * (b_act - a_act)
            psi_now      = self.get_yaw_ref_at(self.current_idx, self.current_t)
            dx = self.state[0] - ref_pt_now[0]
            dy = self.state[1] - ref_pt_now[1]
            e_lat = abs(-np.sin(psi_now) * dx + np.cos(psi_now) * dy)
            e_lon = abs( np.cos(psi_now) * dx + np.sin(psi_now) * dy)
            e_yaw = np.arctan2(np.sin(self.state[2] - psi_now), np.cos(self.state[2] - psi_now))
            v_ref_now = self.get_v_ref_at(self.current_idx, self.current_t)

            self.get_logger().info(
                f'Idx: {self.current_idx}/{n} | '
                f'e_lat: {e_lat:.3f}m | e_lon: {e_lon:.3f}m | '
                f'v: {u[0]:.2f}/{v_ref_now:.2f} m/s | '
                f'δ: {np.degrees(u[1]):.1f}°',
                throttle_duration_sec=0.5
            )

            now_stamp = self.get_clock().now().to_msg()
            self.publish_predicted_path(opt, now_stamp)
            self.publish_debug_metrics(
                self.state[0], self.state[1], self.state[2],
                ref_pt_now[0], ref_pt_now[1], psi_now,
                u[0], v_ref_now, u[1], solve_ms
            )
            self.publish_cmd(u)

        except Exception as e:
            self.get_logger().error(f'MPC failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.publish_cmd(np.zeros(2))

    def publish_cmd(self, u):
        msg                 = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        v, delta            = float(u[0]), float(u[1])
        msg.twist.linear.x  = v
        msg.twist.angular.z = v / self.L * np.tan(delta)
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(AckermannMPC())


if __name__ == '__main__':
    main()