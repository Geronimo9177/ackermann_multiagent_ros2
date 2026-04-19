#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from collections import deque
from std_msgs.msg import Float64MultiArray

import matplotlib.pyplot as plt


class MPCDebugVisualizer(Node):

    def __init__(self):
        super().__init__('mpc_debug_visualizer')

        self.declare_parameter('history_size', 600)
        self.history_size = int(
            self.get_parameter('history_size').get_parameter_value().integer_value
        )

        self.t = deque(maxlen=self.history_size)
        self.x_now = deque(maxlen=self.history_size)
        self.x_ref = deque(maxlen=self.history_size)
        self.y_now = deque(maxlen=self.history_size)
        self.y_ref = deque(maxlen=self.history_size)
        self.yaw_now_deg = deque(maxlen=self.history_size)
        self.yaw_ref_deg = deque(maxlen=self.history_size)
        self.e_lat = deque(maxlen=self.history_size)
        self.e_lon = deque(maxlen=self.history_size)
        self.e_yaw_deg = deque(maxlen=self.history_size)
        self.v_cmd = deque(maxlen=self.history_size)
        self.v_ref = deque(maxlen=self.history_size)
        self.steer_deg = deque(maxlen=self.history_size)
        self.solve_ms = deque(maxlen=self.history_size)

        self.t0 = self.get_clock().now().nanoseconds * 1e-9

        self.create_subscription(Float64MultiArray, '/mpc/debug', self.debug_cb, 30)
        self.create_timer(0.1, self.update_plot)

        self.fig, self.axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
        self.axes = self.axes.flatten()
        self.lines = {
            'x_now': self.axes[0].plot([], [], label='x_now [m]')[0],
            'x_ref': self.axes[0].plot([], [], label='x_ref [m]')[0],
            'y_now': self.axes[1].plot([], [], label='y_now [m]')[0],
            'y_ref': self.axes[1].plot([], [], label='y_ref [m]')[0],
            'yaw_now': self.axes[2].plot([], [], label='yaw_now [deg]')[0],
            'yaw_ref': self.axes[2].plot([], [], label='yaw_ref [deg]')[0],
            'e_lat': self.axes[3].plot([], [], label='e_lat [m]')[0],
            'e_lon': self.axes[3].plot([], [], label='e_lon [m]')[0],
            'e_yaw': self.axes[4].plot([], [], label='e_yaw [deg]')[0],
            'v_cmd': self.axes[5].plot([], [], label='v_cmd [m/s]')[0],
            'v_ref': self.axes[5].plot([], [], label='v_ref [m/s]')[0],
            'steer': self.axes[6].plot([], [], label='steer [deg]')[0],
            'solve_ms': self.axes[7].plot([], [], label='solve [ms]')[0],
        }

        self.axes[0].set_title('X: current vs reference')
        self.axes[1].set_title('Y: current vs reference')
        self.axes[2].set_title('Yaw: current vs reference')
        self.axes[3].set_title('Position errors')
        self.axes[4].set_title('Yaw error')
        self.axes[5].set_title('Velocity command vs reference')
        self.axes[6].set_title('Steering angle')
        self.axes[7].set_title('Solver time')

        self.axes[0].set_ylabel('m')
        self.axes[1].set_ylabel('m')
        self.axes[2].set_ylabel('deg')
        self.axes[3].set_ylabel('m')
        self.axes[4].set_ylabel('deg')
        self.axes[5].set_ylabel('m/s')
        self.axes[6].set_ylabel('deg')
        self.axes[7].set_ylabel('ms')

        self.axes[6].set_xlabel('Time [s]')
        self.axes[7].set_xlabel('Time [s]')

        for ax in self.axes:
            ax.grid(True)
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

        self.get_logger().info('MPC debug visualizer ready. Listening on /mpc/debug')

    def debug_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 11:
            return

        x_now = msg.data[0]
        y_now = msg.data[1]
        yaw_now = msg.data[2]
        x_ref = msg.data[3]
        y_ref = msg.data[4]
        yaw_ref = msg.data[5]

        dx = x_now - x_ref
        dy = y_now - y_ref
        e_lon = np.cos(yaw_ref) * dx + np.sin(yaw_ref) * dy
        e_lat = -np.sin(yaw_ref) * dx + np.cos(yaw_ref) * dy
        e_yaw = np.arctan2(np.sin(yaw_now - yaw_ref), np.cos(yaw_now - yaw_ref))

        now = self.get_clock().now().nanoseconds * 1e-9 - self.t0
        self.t.append(now)
        self.x_now.append(x_now)
        self.x_ref.append(x_ref)
        self.y_now.append(y_now)
        self.y_ref.append(y_ref)
        self.yaw_now_deg.append(np.degrees(yaw_now))
        self.yaw_ref_deg.append(np.degrees(yaw_ref))
        self.e_lat.append(e_lat)
        self.e_lon.append(e_lon)
        self.e_yaw_deg.append(np.degrees(e_yaw))
        self.v_cmd.append(msg.data[6])
        self.v_ref.append(msg.data[7])
        self.steer_deg.append(np.degrees(msg.data[8]))
        self.solve_ms.append(msg.data[10])

    def update_plot(self):
        if len(self.t) < 2:
            return

        tx = np.array(self.t)

        self.lines['x_now'].set_data(tx, np.array(self.x_now))
        self.lines['x_ref'].set_data(tx, np.array(self.x_ref))
        self.lines['y_now'].set_data(tx, np.array(self.y_now))
        self.lines['y_ref'].set_data(tx, np.array(self.y_ref))
        self.lines['yaw_now'].set_data(tx, np.array(self.yaw_now_deg))
        self.lines['yaw_ref'].set_data(tx, np.array(self.yaw_ref_deg))
        self.lines['e_lat'].set_data(tx, np.array(self.e_lat))
        self.lines['e_lon'].set_data(tx, np.array(self.e_lon))
        self.lines['e_yaw'].set_data(tx, np.array(self.e_yaw_deg))
        self.lines['v_cmd'].set_data(tx, np.array(self.v_cmd))
        self.lines['v_ref'].set_data(tx, np.array(self.v_ref))
        self.lines['steer'].set_data(tx, np.array(self.steer_deg))
        self.lines['solve_ms'].set_data(tx, np.array(self.solve_ms))

        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main():
    rclpy.init()
    node = MPCDebugVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()