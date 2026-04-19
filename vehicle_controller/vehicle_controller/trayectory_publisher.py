#!/usr/bin/env python3
# trajectory_publisher.py

import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class TrajectoryPublisher(Node):

    def __init__(self):
        super().__init__('trajectory_publisher')

        self.declare_parameter('trajectory_file', 'baylands_nurbs.csv')

        self.pub   = self.create_publisher(Path, '/trajectory', 10)
        self.timer = self.create_timer(1.0, self.publish_path)

        pkg_path = get_package_share_directory('vehicle_controller')
        filename = self.get_parameter('trajectory_file').value
        csv_file = os.path.join(pkg_path, 'trajectories', filename)

        data      = np.loadtxt(csv_file, delimiter=",", skiprows=1)
        self.x    = data[:, 0]
        self.y    = data[:, 1]

        self.get_logger().info(
            f'Loaded {len(self.x)} control points from {filename}'
        )

    def publish_path(self):
        path                 = Path()
        path.header.frame_id = 'odom'
        path.header.stamp    = self.get_clock().now().to_msg()

        for i in range(len(self.x)):
            pose                    = PoseStamped()
            pose.header.frame_id    = 'odom'
            pose.header.stamp       = path.header.stamp
            pose.pose.position.x    = float(self.x[i])
            pose.pose.position.y    = float(self.y[i])
            pose.pose.position.z    = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.pub.publish(path)
        self.get_logger().info(f'Published {len(path.poses)} control points')
        raise SystemExit


def main():
    rclpy.init()
    node = TrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()