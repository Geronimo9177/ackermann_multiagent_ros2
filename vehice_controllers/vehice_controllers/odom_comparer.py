#!/usr/bin/env python3
import sys
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from message_filters import Subscriber, ApproximateTimeSynchronizer

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore


class OdometryComparer(Node):
    def __init__(self):
        super().__init__('odom_comparer')

        # Parameters
        self.declare_parameter('window_size', 10000)
        self.window_size = self.get_parameter('window_size').value

        # --- Time synchronized subscribers ---
        self.gt_sub = Subscriber(self, Odometry, '/ground_truth_odom')
        self.ctrl_sub = Subscriber(self, Odometry, '/odometry/filtered')

        self.sync = ApproximateTimeSynchronizer(
            [self.gt_sub, self.ctrl_sub],
            queue_size=20,
            slop=0.03
        )
        self.sync.registerCallback(self.synced_callback)

        # Storage
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

        self.errors = {
            'position': deque(maxlen=self.window_size),
            'x': deque(maxlen=self.window_size),
            'y': deque(maxlen=self.window_size),
            'yaw': deque(maxlen=self.window_size),
            'vx': deque(maxlen=self.window_size),
            'vyaw': deque(maxlen=self.window_size),
            'time': deque(maxlen=self.window_size)
        }

        self.init_plots()

        self.get_logger().info('PyQtGraph Odometry Comparer started (time synced)')

    # ---------------- Math utils ----------------

    def quaternion_to_yaw(self, quat):
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    # ---------------- Plot setup ----------------

    def init_plots(self):
        self.app = QtWidgets.QApplication(sys.argv)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle('Odometry Comparison PyQtGraph')
        self.win.resize(1400, 900)

        # ---- Row 1 ----
        self.ax1 = self.win.addPlot(title='2D Trajectory')
        self.ax1.setAspectLocked(True)
        self.gt_traj = self.ax1.plot(pen='g')
        self.ctrl_traj = self.ax1.plot(pen='b')

        self.axYawErr = self.win.addPlot(title='Yaw Error (deg)')
        self.yaw_err_curve = self.axYawErr.plot(pen='m')

        self.win.nextRow()

        # ---- Row 2 ----
        self.axErrX = self.win.addPlot(title='Position Error X')
        self.err_x_curve = self.axErrX.plot(pen='r')

        self.axErrY = self.win.addPlot(title='Position Error Y')
        self.err_y_curve = self.axErrY.plot(pen='b')

        self.win.nextRow()

        # ---- Row 3 ----
        self.ax4 = self.win.addPlot(title='Linear Velocity')
        self.gt_vx_curve = self.ax4.plot(pen='g')
        self.ctrl_vx_curve = self.ax4.plot(pen='b')

        self.ax5 = self.win.addPlot(title='Angular Velocity (deg/s)')
        self.gt_vyaw_curve = self.ax5.plot(pen='g')
        self.ctrl_vyaw_curve = self.ax5.plot(pen='b')

        self.win.nextRow()

        # ---- Row 4 ----
        self.axVxErr = self.win.addPlot(title='Linear Velocity Error')
        self.vx_err_curve = self.axVxErr.plot(pen='r')

        self.axVyawErr = self.win.addPlot(title='Angular Velocity Error (deg/s)')
        self.vyaw_err_curve = self.axVyawErr.plot(pen='m')

        # Plot timer (~50 Hz)
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update)
        self.plot_timer.start(20)

        self.ros_timer = QtCore.QTimer()
        self.ros_timer.timeout.connect(self.spin_ros)
        self.ros_timer.start(1)

    # ---------------- ROS spin ----------------

    def spin_ros(self):
        if not rclpy.ok():
            return
        try:
            rclpy.spin_once(self, timeout_sec=0)
        except Exception:
            pass

    # ---------------- SYNCHRONIZED CALLBACK ----------------

    def synced_callback(self, gt_msg, ctrl_msg):

        t = (
            gt_msg.header.stamp.sec +
            gt_msg.header.stamp.nanosec * 1e-9
        )

        # Ground truth
        gt_x = gt_msg.pose.pose.position.x
        gt_y = gt_msg.pose.pose.position.y
        gt_yaw = self.quaternion_to_yaw(gt_msg.pose.pose.orientation)
        gt_vx = gt_msg.twist.twist.linear.x
        gt_vyaw = gt_msg.twist.twist.angular.z

        # Filtered
        ctrl_x = ctrl_msg.pose.pose.position.x
        ctrl_y = ctrl_msg.pose.pose.position.y
        ctrl_yaw = self.quaternion_to_yaw(ctrl_msg.pose.pose.orientation)
        ctrl_vx = ctrl_msg.twist.twist.linear.x
        ctrl_vyaw = ctrl_msg.twist.twist.angular.z

        # Save data
        self.gt_data['x'].append(gt_x)
        self.gt_data['y'].append(gt_y)
        self.gt_data['vx'].append(gt_vx)
        self.gt_data['vyaw'].append(gt_vyaw)
        self.gt_data['time'].append(t)

        self.ctrl_data['x'].append(ctrl_x)
        self.ctrl_data['y'].append(ctrl_y)
        self.ctrl_data['vx'].append(ctrl_vx)
        self.ctrl_data['vyaw'].append(ctrl_vyaw)
        self.ctrl_data['time'].append(t)

        # Errors
        err_x = ctrl_x - gt_x
        err_y = ctrl_y - gt_y
        pos_err = math.sqrt(err_x**2 + err_y**2)
        yaw_err = self.normalize_angle(ctrl_yaw - gt_yaw)
        vx_err = ctrl_vx - gt_vx
        vyaw_err = ctrl_vyaw - gt_vyaw

        self.errors['x'].append(err_x)
        self.errors['y'].append(err_y)
        self.errors['position'].append(pos_err)
        self.errors['yaw'].append(yaw_err)
        self.errors['vx'].append(vx_err)
        self.errors['vyaw'].append(vyaw_err)
        self.errors['time'].append(t)

    # ---------------- Plot update ----------------

    def update(self):
        if len(self.errors['time']) == 0:
            return

        self.gt_traj.setData(self.gt_data['x'], self.gt_data['y'])
        self.ctrl_traj.setData(self.ctrl_data['x'], self.ctrl_data['y'])

        self.yaw_err_curve.setData(
            self.errors['time'],
            np.degrees(self.errors['yaw'])
        )

        self.err_x_curve.setData(self.errors['time'], self.errors['x'])
        self.err_y_curve.setData(self.errors['time'], self.errors['y'])

        self.gt_vx_curve.setData(self.gt_data['time'], self.gt_data['vx'])
        self.ctrl_vx_curve.setData(self.ctrl_data['time'], self.ctrl_data['vx'])

        self.gt_vyaw_curve.setData(
            self.gt_data['time'],
            np.degrees(self.gt_data['vyaw'])
        )
        self.ctrl_vyaw_curve.setData(
            self.ctrl_data['time'],
            np.degrees(self.ctrl_data['vyaw'])
        )

        self.vx_err_curve.setData(self.errors['time'], self.errors['vx'])
        self.vyaw_err_curve.setData(
            self.errors['time'],
            np.degrees(self.errors['vyaw'])
        )

    def run(self):
        import signal

        signal.signal(signal.SIGINT, lambda *args: self.app.quit())
        self.app.aboutToQuit.connect(lambda: (rclpy.ok() and rclpy.shutdown()))
        self.app.exec()


def main():
    rclpy.init()
    node = OdometryComparer()
    try:
        node.run()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
