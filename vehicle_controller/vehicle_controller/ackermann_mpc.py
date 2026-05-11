#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import casadi as ca
import time

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Float64MultiArray
from tf_transformations import quaternion_matrix


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


def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def nearest_angle(reference, around):
    """Map reference angle to the 2*pi branch closest to around."""
    return float(around + wrap_angle(reference - around))


def _yaw_from_pose(pose_stamped):
    q = pose_stamped.pose.orientation
    R = quaternion_matrix([q.x, q.y, q.z, q.w])
    forward = R[:3, 0]
    fx, fy = forward[0], forward[1]
    norm = np.hypot(fx, fy)
    if norm < 1e-6:
        return 0.0
    return float(np.arctan2(fy, fx))


class AckermannMPC(Node):

    def __init__(self):
        super().__init__('ackermann_mpc')

        # MPC parameters
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('N', 20)
        self.declare_parameter('wheelbase', 2.55)
        self.declare_parameter('max_steer', 0.6458)
        self.declare_parameter('max_speed', 8.0)

        self.dt        = self.get_parameter('dt').value
        self.N         = self.get_parameter('N').value
        self.L         = self.get_parameter('wheelbase').value
        self.max_steer = self.get_parameter('max_steer').value
        self.max_speed = self.get_parameter('max_speed').value

        # Internal state (from the local EKF)
        self.state         = np.zeros(3)
        self.prev_u        = np.zeros(2)
        self.last_solution = None
        self.yaw_cont      = None

        # Drift offset: difference between fused and local in the odom frame
        # Applied to the path so the MPC sees corrected waypoints
        # without  modifying the MPC state (self.state)
        self.drift_offset = np.zeros(2)   # [dx, dy]

        # Arrays from the MultiArray message (TOPP node)
        self.xy_arr  = None   # (N, 2) (map frame)
        self.yaw_arr = None   # (N,)
        self.v_arr   = None   # (N,)
        self.s_arr   = None   # (N,)

        # Path corrected by drift (odom frame)
        self.xy_arr_odom = None

        self.odom_received  = False
        self.fused_received = False
        self.path_received  = False

        # Path tracking
        self.current_idx   = None
        self.current_t     = 0.0
        self.search_window = 10
        self.s_search      = 3

        # Goal completion criteria
        self.goal_tolerance  = 0.10
        self.route_completed = False

        # ── Subscribers ──────────────────────────────────────────────
        self.create_subscription(Odometry, '/odometry/local',
                                 self.odom_cb, 10)

        self.create_subscription(Odometry, '/odometry/fused',
                                 self.fused_cb, 10)

        self.create_subscription(Float64MultiArray, '/trajectory_topp',
                                 self.trajectory_cb, 10)

        # ── Publishers ───────────────────────────────────────────────
        self.cmd_pub            = self.create_publisher(TwistStamped,      '/cmd_vel',             10)
        self.debug_pub          = self.create_publisher(Float64MultiArray,  '/mpc/debug',           10)
        self.predicted_path_pub = self.create_publisher(Path,               '/mpc/predicted_path',  10)

        self.get_logger().info('Setting up MPC solver...')
        self.setup_mpc()
        self.get_logger().info('MPC solver ready!')

        self.create_timer(self.dt, self.control_loop)
        self.create_timer(2.0,     self.debug_status)

    def debug_status(self):
        self.get_logger().info(
            f'Odom: {self.odom_received} | Fused: {self.fused_received} | '
            f'Path: {self.path_received} | Idx: {self.current_idx} | '
            f'State: [{self.state[0]:.2f}, {self.state[1]:.2f}, '
            f'{np.degrees(self.state[2]):.1f}°] | '
            f'Drift offset: [{self.drift_offset[0]:.3f}, {self.drift_offset[1]:.3f}]'
        )

    # ── MPC setup ────────────────────────────────────────────────────
    def setup_mpc(self):
        nx, nu = 3, 2
        n_ref  = 4

        X = ca.MX.sym('X', nx, self.N + 1)
        U = ca.MX.sym('U', nu, self.N)
        P = ca.MX.sym('P', nx + n_ref * self.N)

        Q_lon = 3.0
        Q_lat = 25.0
        Q_yaw = 25.0
        Q_v   = 1.0

        R  = np.diag([0.1, 0.6])
        Rd = np.diag([1.5, 50.0])

        cost = 0
        g    = []

        for k in range(self.N):
            st  = X[:, k]
            con = U[:, k]
            ref = P[nx + n_ref * k : nx + n_ref * k + n_ref]

            dx  = st[0] - ref[0]
            dy  = st[1] - ref[1]
            psi = ref[2]

            e_s   =  ca.cos(psi) * dx + ca.sin(psi) * dy
            e_l   = -ca.sin(psi) * dx + ca.cos(psi) * dy
            e_yaw = ca.atan2(ca.sin(st[2] - ref[2]), ca.cos(st[2] - ref[2]))
            e_v   = con[0] - ref[3]

            cost += Q_lon * e_s**2
            cost += Q_lat * e_l**2
            cost += Q_yaw * e_yaw**2
            cost += Q_v   * e_v**2
            cost += ca.mtimes([con.T, R, con])

            if k > 0:
                du    = U[:, k] - U[:, k - 1]
                cost += ca.mtimes([du.T, Rd, du])

            x_next = ca.vertcat(
                st[0] + con[0] * ca.cos(st[2]) * self.dt,
                st[1] + con[0] * ca.sin(st[2]) * self.dt,
                st[2] + (con[0] / self.L) * ca.tan(con[1]) * self.dt
            )
            g.append(X[:, k + 1] - x_next)

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

    # ── Callbacks ─────────────────────────────────────────────────────

    def odom_cb(self, msg):
        """Local EKF state used by the MPC."""
        p           = msg.pose.pose.position
        yaw_wrapped = _yaw_from_pose(msg.pose)

        if self.yaw_cont is None:
            self.yaw_cont = yaw_wrapped
        else:
            dyaw = wrap_angle(yaw_wrapped - wrap_angle(self.yaw_cont))
            self.yaw_cont += dyaw

        self.state = np.array([p.x, p.y, self.yaw_cont])

        if not self.odom_received:
            self.get_logger().info('First odometry/local received!')
            self.odom_received = True

    def fused_cb(self, msg):
        """OdometryFusion output updates drift_offset only. """
        if not self.odom_received:
            return

        fx = msg.pose.pose.position.x
        fy = msg.pose.pose.position.y

        # Offset between the fused position and the current local position
        new_offset = np.array([fx - self.state[0],
                                fy - self.state[1]])

        # Smooth the offset to avoid abrupt path jumps
        alpha = 0.1
        self.drift_offset = (1.0 - alpha) * self.drift_offset + alpha * new_offset

        # Recompute the path in the odom frame with the new offset
        if self.xy_arr is not None:
            self._reproject_path()

        if not self.fused_received:
            self.get_logger().info(
                f'First odometry/fused received! '
                f'offset=[{self.drift_offset[0]:.3f}, {self.drift_offset[1]:.3f}]'
            )
            self.fused_received = True

    def trajectory_cb(self, msg: Float64MultiArray):
        """Receive the path in the map frame and project it to odom frame."""
        n      = msg.layout.dim[0].size
        fields = msg.layout.dim[1].size

        data = np.array(msg.data).reshape(n, fields)
        self.xy_arr  = data[:, 0:2]   # map frame
        self.yaw_arr = data[:, 2]
        self.v_arr   = data[:, 3]
        self.s_arr   = data[:, 4]

        self.current_idx     = None
        self.current_t       = 0.0
        self.route_completed = False
        self.last_solution   = None

        # Project the path to the odom frame using the current drift_offset
        self._reproject_path()

        if not self.path_received:
            self.get_logger().info(
                f'Trajectory TOPP: {n} pts | '
                f'v [{self.v_arr.min():.2f}, {self.v_arr.max():.2f}] m/s'
            )
            self.path_received = True

    def _reproject_path(self):
        """Transform the path from map frame to odom frame by subtracting drift_offset.
        
        frame_odom = frame_map - drift_offset
        
        drift_offset is the accumulated difference between the fused position
        (which includes GPS correction) and the local position (pure odom/IMU).
        """
        if self.xy_arr is None:
            return
        # The path in odom is the map path minus the drift offset
        self.xy_arr_odom = self.xy_arr - self.drift_offset[np.newaxis, :]

    # ── Geometry helpers ──────────────────────────────────────────────

    def get_segment(self, i, n):
        idx_a = min(i, n - 2)
        # Use the path projected into odom
        return self.xy_arr_odom[idx_a], self.xy_arr_odom[idx_a + 1]

    def get_v_ref_at(self, idx, t):
        idx_a = min(idx, len(self.v_arr) - 2)
        return float((1.0 - t) * self.v_arr[idx_a] + t * self.v_arr[idx_a + 1])

    def get_yaw_ref_at(self, idx, t):
        idx_a = min(idx, len(self.yaw_arr) - 2)
        a0, a1 = self.yaw_arr[idx_a], self.yaw_arr[idx_a + 1]
        return float(a0 + t * np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)))

    # ── Index update ──────────────────────────────────────────────────

    def update_current_index(self):
        if self.current_idx is None:
            self.current_idx = 0
            self.current_t   = 0.0
            return

        n = len(self.xy_arr_odom)
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

    # ── Warm start ────────────────────────────────────────────────────

    def get_warm_start(self, opt):
        nx_states      = 3 * (self.N + 1)
        states         = opt[:nx_states].reshape(self.N + 1, 3)
        inputs         = opt[nx_states:].reshape(self.N, 2)
        states_shifted = np.roll(states, -1, axis=0); states_shifted[-1] = states[-1]
        inputs_shifted = np.roll(inputs, -1, axis=0); inputs_shifted[-1] = inputs[-1]
        return np.concatenate([states_shifted.flatten(), inputs_shifted.flatten()])

    def should_complete_route(self):
        if self.xy_arr_odom is None or self.current_idx is None:
            return False
        goal          = self.xy_arr_odom[-1]
        goal_distance = float(np.hypot(self.state[0] - goal[0],
                                       self.state[1] - goal[1]))
        return goal_distance <= self.goal_tolerance

    def make_pose_stamped(self, x, y, yaw, stamp):
        pose = PoseStamped()
        pose.header.frame_id      = 'odom'
        pose.header.stamp         = stamp
        pose.pose.position.x      = float(x)
        pose.pose.position.y      = float(y)
        pose.pose.position.z      = 0.0
        pose.pose.orientation.z   = float(np.sin(yaw * 0.5))
        pose.pose.orientation.w   = float(np.cos(yaw * 0.5))
        return pose

    def publish_predicted_path(self, opt, stamp):
        nx_states = 3 * (self.N + 1)
        states    = opt[:nx_states].reshape(self.N + 1, 3)
        msg               = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp    = stamp
        msg.poses = [self.make_pose_stamped(x, y, yaw, stamp) for x, y, yaw in states]
        self.predicted_path_pub.publish(msg)

    def publish_debug_metrics(self, x_now, y_now, yaw_now, x_ref, y_ref, yaw_ref,
                              v_cmd, v_ref, steer, solve_ms):
        msg      = Float64MultiArray()
        msg.data = [
            float(x_now),   float(y_now),   float(yaw_now),
            float(x_ref),   float(y_ref),   float(yaw_ref),
            float(v_cmd),   float(v_ref),   float(steer),
            float(self.current_idx if self.current_idx is not None else -1),
            float(solve_ms),
            float(self.drift_offset[0]),   # extra: x offset for debug
            float(self.drift_offset[1]),   # extra: y offset for debug
        ]
        self.debug_pub.publish(msg)

    # ── Control loop ──────────────────────────────────────────────────

    def control_loop(self):
        if self.xy_arr_odom is None or not self.odom_received or self.route_completed:
            return

        n = len(self.xy_arr_odom)
        if n < 2:
            self.get_logger().warn('Trajectory too short!', throttle_duration_sec=2.0)
            self.publish_cmd(np.zeros(2))
            return

        try:
            self.update_current_index()

            if self.current_idx >= int(0.95 * (n - 1)):
                if self.should_complete_route():
                    self.route_completed = True
                    self.path_received   = False
                    self.current_idx     = None
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
            yaw_anchor = self.state[2]

            for k in range(self.N):
                if seg_len > 0.001:
                    ref_pt      = a + path_t * (b - a)
                    ref_yaw_raw = self.get_yaw_ref_at(path_idx, path_t)
                    ref_yaw     = nearest_angle(ref_yaw_raw, yaw_anchor)
                else:
                    ref_pt  = b
                    ref_yaw = self.state[2]

                yaw_anchor = ref_yaw
                v_ref_k    = self.get_v_ref_at(path_idx, path_t)

                ref.extend([ref_pt[0], ref_pt[1], ref_yaw, v_ref_k])

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

            t0       = time.perf_counter()
            sol      = self.solver(x0=x0, p=P_vec, lbg=0, ubg=0,
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

            # Logging in the odom frame
            a_act, b_act = self.get_segment(self.current_idx, n)
            ref_pt_now   = a_act + self.current_t * (b_act - a_act)
            psi_now_raw  = self.get_yaw_ref_at(self.current_idx, self.current_t)
            psi_now      = nearest_angle(psi_now_raw, self.state[2])
            dx    = self.state[0] - ref_pt_now[0]
            dy    = self.state[1] - ref_pt_now[1]
            e_lat = abs(-np.sin(psi_now) * dx + np.cos(psi_now) * dy)
            e_lon = abs( np.cos(psi_now) * dx + np.sin(psi_now) * dy)
            e_yaw = np.arctan2(np.sin(self.state[2] - psi_now),
                               np.cos(self.state[2] - psi_now))
            v_ref_now = self.get_v_ref_at(self.current_idx, self.current_t)

            self.get_logger().info(
                f'Idx: {self.current_idx}/{n} | '
                f'e_lat: {e_lat:.3f}m | e_lon: {e_lon:.3f}m | '
                f'e_yaw: {np.degrees(e_yaw):.1f}° | '
                f'v: {u[0]:.2f}/{v_ref_now:.2f} m/s | '
                f'δ: {np.degrees(u[1]):.1f}° | '
                f'drift: [{self.drift_offset[0]:.3f}, {self.drift_offset[1]:.3f}]',
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