#!/usr/bin/env python3
import math
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.msg import MultiDOFCommand, MultiDOFStateStamped
import matplotlib.pyplot as plt


class JointControllerPlotter(Node):
    def __init__(self):
        super().__init__('joint_controller_plotter')

        self.declare_parameter('window_size', 1000)
        self.declare_parameter('update_rate', 100.0)
        self.declare_parameter('steering_topic', '/steering_pid/reference')
        self.declare_parameter('rear_wheel_topic', '/rear_wheel_pid/reference')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('rear_wheel_state_topic', '/rear_wheel_pid/controller_state')
        self.declare_parameter('left_steer_joint', 'left_steering_joint')
        self.declare_parameter('right_steer_joint', 'right_steering_joint')
        self.declare_parameter('left_rear_joint', 'left_rear_axle')
        self.declare_parameter('right_rear_joint', 'right_rear_axle')

        self.window_size = self.get_parameter('window_size').value
        update_rate = self.get_parameter('update_rate').value
        steering_topic = self.get_parameter('steering_topic').value
        rear_wheel_topic = self.get_parameter('rear_wheel_topic').value
        joint_states_topic = self.get_parameter('joint_states_topic').value
        rear_wheel_state_topic = self.get_parameter('rear_wheel_state_topic').value

        self.left_steer_joint = self.get_parameter('left_steer_joint').value
        self.right_steer_joint = self.get_parameter('right_steer_joint').value
        self.left_rear_joint = self.get_parameter('left_rear_joint').value
        self.right_rear_joint = self.get_parameter('right_rear_joint').value

        self._lock = threading.Lock()
        self.start_time = self.get_clock().now()

        self.time = deque(maxlen=self.window_size)

        self.ref = {
            self.left_steer_joint: {'pos': deque(maxlen=self.window_size), 'vel': deque(maxlen=self.window_size)},
            self.right_steer_joint: {'pos': deque(maxlen=self.window_size), 'vel': deque(maxlen=self.window_size)},
            self.left_rear_joint: {'pos': deque(maxlen=self.window_size), 'vel': deque(maxlen=self.window_size)},
            self.right_rear_joint: {'pos': deque(maxlen=self.window_size), 'vel': deque(maxlen=self.window_size)},
        }
        self.act = {
            self.left_steer_joint: {'pos': deque(maxlen=self.window_size), 'vel': deque(maxlen=self.window_size)},
            self.right_steer_joint: {'pos': deque(maxlen=self.window_size), 'vel': deque(maxlen=self.window_size)},
            self.left_rear_joint: {'pos': deque(maxlen=self.window_size), 'vel': deque(maxlen=self.window_size)},
            self.right_rear_joint: {'pos': deque(maxlen=self.window_size), 'vel': deque(maxlen=self.window_size)},
        }

        self.pid_left_error = deque(maxlen=self.window_size)
        self.pid_right_error = deque(maxlen=self.window_size)
        self.pid_left_output = deque(maxlen=self.window_size)
        self.pid_right_output = deque(maxlen=self.window_size)

        self.joint_state_sub = self.create_subscription(
            JointState,
            joint_states_topic,
            self.joint_state_callback,
            50,
        )

        self.steering_sub = self.create_subscription(
            MultiDOFCommand,
            steering_topic,
            self.steering_traj_callback,
            10,
        )

        self.rear_sub = self.create_subscription(
            MultiDOFCommand,
            rear_wheel_topic,
            self.rear_traj_callback,
            10,
        )

        self.rear_state_sub = self.create_subscription(
            MultiDOFStateStamped,
            rear_wheel_state_topic,
            self.rear_state_callback,
            10,
        )

        plt.ion()
        self.setup_plots()
        plt.show(block=False)

        self.plot_timer = self.create_timer(1.0 / update_rate, self.update_plots)

        self.get_logger().info('Joint Controller Plotter started')

    def joint_state_callback(self, msg: JointState):
        now = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        with self._lock:
            self.time.append(now)
            for name in self.act.keys():
                if name in msg.name:
                    idx = msg.name.index(name)
                    if idx < len(msg.position):
                        self.act[name]['pos'].append(msg.position[idx])
                    if idx < len(msg.velocity):
                        self.act[name]['vel'].append(msg.velocity[idx])
                else:
                    if len(self.act[name]['pos']) < len(self.time):
                        self.act[name]['pos'].append(self.act[name]['pos'][-1] if self.act[name]['pos'] else 0.0)
                    if len(self.act[name]['vel']) < len(self.time):
                        self.act[name]['vel'].append(self.act[name]['vel'][-1] if self.act[name]['vel'] else 0.0)

            for name in self.ref.keys():
                if len(self.ref[name]['pos']) < len(self.time):
                    self.ref[name]['pos'].append(self.ref[name]['pos'][-1] if self.ref[name]['pos'] else 0.0)
                if len(self.ref[name]['vel']) < len(self.time):
                    self.ref[name]['vel'].append(self.ref[name]['vel'][-1] if self.ref[name]['vel'] else 0.0)

            if len(self.pid_left_error) < len(self.time):
                self.pid_left_error.append(self.pid_left_error[-1] if self.pid_left_error else 0.0)
            if len(self.pid_right_error) < len(self.time):
                self.pid_right_error.append(self.pid_right_error[-1] if self.pid_right_error else 0.0)
            if len(self.pid_left_output) < len(self.time):
                self.pid_left_output.append(self.pid_left_output[-1] if self.pid_left_output else 0.0)
            if len(self.pid_right_output) < len(self.time):
                self.pid_right_output.append(self.pid_right_output[-1] if self.pid_right_output else 0.0)

    def steering_traj_callback(self, msg: MultiDOFCommand):
        self._handle_steering(msg)

    def rear_traj_callback(self, msg: MultiDOFCommand):
        self._handle_rear(msg)

    def rear_state_callback(self, msg: MultiDOFStateStamped):
        with self._lock:
            for dof_state in msg.dof_states:
                if dof_state.name == self.left_rear_joint:
                    self.pid_left_error.append(dof_state.error)
                    self.pid_left_output.append(dof_state.output)
                elif dof_state.name == self.right_rear_joint:
                    self.pid_right_error.append(dof_state.error)
                    self.pid_right_output.append(dof_state.output)

    def _handle_steering(self, msg: MultiDOFCommand):
        with self._lock:
            if len(msg.dof_names) == 0:
                return
            name_to_index = {name: i for i, name in enumerate(msg.dof_names)}
            if self.left_steer_joint in name_to_index and self.right_steer_joint in name_to_index:
                li = name_to_index[self.left_steer_joint]
                ri = name_to_index[self.right_steer_joint]
                if li < len(msg.values):
                    self.ref[self.left_steer_joint]['pos'].append(msg.values[li])
                if ri < len(msg.values):
                    self.ref[self.right_steer_joint]['pos'].append(msg.values[ri])

    def _handle_rear(self, msg: MultiDOFCommand):
        with self._lock:
            if len(msg.dof_names) == 0:
                return
            name_to_index = {name: i for i, name in enumerate(msg.dof_names)}
            if self.left_rear_joint in name_to_index and self.right_rear_joint in name_to_index:
                li = name_to_index[self.left_rear_joint]
                ri = name_to_index[self.right_rear_joint]
                if li < len(msg.values):
                    self.ref[self.left_rear_joint]['vel'].append(msg.values[li])
                if ri < len(msg.values):
                    self.ref[self.right_rear_joint]['vel'].append(msg.values[ri])

    def setup_plots(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.suptitle('Joint Reference vs Actual', fontsize=16, fontweight='bold')

        self.ax1 = plt.subplot(2, 3, 1)
        self.ax1.set_title('Left Steering (pos)')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('deg')
        self.ax1.grid(True, alpha=0.3)
        self.l1_ref, = self.ax1.plot([], [], 'r--', label='ref')
        self.l1_act, = self.ax1.plot([], [], 'b-', label='actual')
        self.ax1.legend()

        self.ax2 = plt.subplot(2, 3, 2)
        self.ax2.set_title('Right Steering (pos)')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('deg')
        self.ax2.grid(True, alpha=0.3)
        self.l2_ref, = self.ax2.plot([], [], 'r--', label='ref')
        self.l2_act, = self.ax2.plot([], [], 'b-', label='actual')
        self.ax2.legend()

        self.ax3 = plt.subplot(2, 3, 3)
        self.ax3.set_title('Rear Wheels (vel)')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('rad/s')
        self.ax3.grid(True, alpha=0.3)
        self.l3_ref_l, = self.ax3.plot([], [], 'r--', label='left ref')
        self.l3_act_l, = self.ax3.plot([], [], 'b-', label='left actual')
        self.l3_ref_r, = self.ax3.plot([], [], 'm--', label='right ref')
        self.l3_act_r, = self.ax3.plot([], [], 'g-', label='right actual')
        self.ax3.legend()

        self.ax4 = plt.subplot(2, 3, 4)
        self.ax4.set_title('Steering Error (pos)')
        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('deg')
        self.ax4.grid(True, alpha=0.3)
        self.l4_l, = self.ax4.plot([], [], 'b-', label='left error')
        self.l4_r, = self.ax4.plot([], [], 'g-', label='right error')
        self.ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax4.legend()

        self.ax5 = plt.subplot(2, 3, 5)
        self.ax5.set_title('Rear Wheel Error (vel)')
        self.ax5.set_xlabel('Time (s)')
        self.ax5.set_ylabel('rad/s')
        self.ax5.grid(True, alpha=0.3)
        self.l5_l, = self.ax5.plot([], [], 'b-', label='left error')
        self.l5_r, = self.ax5.plot([], [], 'g-', label='right error')
        self.ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax5.legend()

        self.ax6 = plt.subplot(2, 3, 6)
        self.ax6.set_title('PID Output (rear wheels)')
        self.ax6.set_xlabel('Time (s)')
        self.ax6.set_ylabel('output')
        self.ax6.grid(True, alpha=0.3)
        self.l6_out_l, = self.ax6.plot([], [], 'b-', label='left output')
        self.l6_out_r, = self.ax6.plot([], [], 'g-', label='right output')
        self.ax6.legend()

        plt.tight_layout()

    def update_plots(self):
        with self._lock:
            if len(self.time) < 2:
                return

            t = list(self.time)
            ls_ref = list(self.ref[self.left_steer_joint]['pos'])
            ls_act = list(self.act[self.left_steer_joint]['pos'])
            rs_ref = list(self.ref[self.right_steer_joint]['pos'])
            rs_act = list(self.act[self.right_steer_joint]['pos'])

            lr_ref_v = list(self.ref[self.left_rear_joint]['vel'])
            rr_ref_v = list(self.ref[self.right_rear_joint]['vel'])
            lr_act_v = list(self.act[self.left_rear_joint]['vel'])
            rr_act_v = list(self.act[self.right_rear_joint]['vel'])

            pid_left_error = list(self.pid_left_error)
            pid_right_error = list(self.pid_right_error)
            pid_left_output = list(self.pid_left_output)
            pid_right_output = list(self.pid_right_output)

            min_len = min(
                len(t),
                len(ls_ref), len(ls_act), len(rs_ref), len(rs_act),
                len(lr_ref_v), len(rr_ref_v), len(lr_act_v), len(rr_act_v),
                len(pid_left_error), len(pid_right_error), len(pid_left_output), len(pid_right_output),
            )
            if min_len < 2:
                return

            t = t[-min_len:]
            ls_ref = ls_ref[-min_len:]
            ls_act = ls_act[-min_len:]
            rs_ref = rs_ref[-min_len:]
            rs_act = rs_act[-min_len:]
            lr_ref_v = lr_ref_v[-min_len:]
            rr_ref_v = rr_ref_v[-min_len:]
            lr_act_v = lr_act_v[-min_len:]
            rr_act_v = rr_act_v[-min_len:]
            pid_left_error = pid_left_error[-min_len:]
            pid_right_error = pid_right_error[-min_len:]
            pid_left_output = pid_left_output[-min_len:]
            pid_right_output = pid_right_output[-min_len:]

        ls_ref_deg = [math.degrees(v) for v in ls_ref]
        ls_act_deg = [math.degrees(v) for v in ls_act]
        self.l1_ref.set_data(t, ls_ref_deg)
        self.l1_act.set_data(t, ls_act_deg)
        self.ax1.relim()
        self.ax1.autoscale_view()

        rs_ref_deg = [math.degrees(v) for v in rs_ref]
        rs_act_deg = [math.degrees(v) for v in rs_act]
        self.l2_ref.set_data(t, rs_ref_deg)
        self.l2_act.set_data(t, rs_act_deg)
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.l3_ref_l.set_data(t, lr_ref_v)
        self.l3_act_l.set_data(t, lr_act_v)
        self.l3_ref_r.set_data(t, rr_ref_v)
        self.l3_act_r.set_data(t, rr_act_v)
        self.ax3.relim()
        self.ax3.autoscale_view()

        steer_err_l = [math.degrees(a - r) for a, r in zip(ls_act, ls_ref)]
        steer_err_r = [math.degrees(a - r) for a, r in zip(rs_act, rs_ref)]
        self.l4_l.set_data(t, steer_err_l)
        self.l4_r.set_data(t, steer_err_r)
        self.ax4.relim()
        self.ax4.autoscale_view()

        rear_err_l = [a - r for a, r in zip(lr_act_v, lr_ref_v)]
        rear_err_r = [a - r for a, r in zip(rr_act_v, rr_ref_v)]
        self.l5_l.set_data(t, rear_err_l)
        self.l5_r.set_data(t, rear_err_r)
        self.ax5.relim()
        self.ax5.autoscale_view()

        self.l6_out_l.set_data(t, pid_left_output)
        self.l6_out_r.set_data(t, pid_right_output)
        self.ax6.relim()
        self.ax6.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = JointControllerPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
