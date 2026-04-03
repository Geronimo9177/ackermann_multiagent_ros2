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

        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        self.lines = {
            'e_lat': self.axes[0].plot([], [], label='e_lat [m]')[0],
            'e_lon': self.axes[0].plot([], [], label='e_lon [m]')[0],
            'e_yaw': self.axes[1].plot([], [], label='e_yaw [deg]')[0],
            'v_cmd': self.axes[2].plot([], [], label='v_cmd [m/s]')[0],
            'v_ref': self.axes[2].plot([], [], label='v_ref [m/s]')[0],
            'steer': self.axes[2].plot([], [], label='steer [deg]')[0],
        }

        self.axes[0].set_ylabel('Position error')
        self.axes[1].set_ylabel('Yaw error')
        self.axes[2].set_ylabel('Control')
        self.axes[2].set_xlabel('Time [s]')

        for ax in self.axes:
            ax.grid(True)
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

        self.get_logger().info('MPC debug visualizer ready. Listening on /mpc/debug')

    def debug_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 8:
            return

        now = self.get_clock().now().nanoseconds * 1e-9 - self.t0
        self.t.append(now)
        self.e_lat.append(msg.data[0])
        self.e_lon.append(msg.data[1])
        self.e_yaw_deg.append(np.degrees(msg.data[2]))
        self.v_cmd.append(msg.data[3])
        self.v_ref.append(msg.data[4])
        self.steer_deg.append(np.degrees(msg.data[5]))
        self.solve_ms.append(msg.data[7])

    def update_plot(self):
        if len(self.t) < 2:
            return

        tx = np.array(self.t)

        self.lines['e_lat'].set_data(tx, np.array(self.e_lat))
        self.lines['e_lon'].set_data(tx, np.array(self.e_lon))
        self.lines['e_yaw'].set_data(tx, np.array(self.e_yaw_deg))
        self.lines['v_cmd'].set_data(tx, np.array(self.v_cmd))
        self.lines['v_ref'].set_data(tx, np.array(self.v_ref))
        self.lines['steer'].set_data(tx, np.array(self.steer_deg))

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