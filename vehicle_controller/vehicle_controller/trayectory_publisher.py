#!/usr/bin/env python3
# trajectory_publisher.py

import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class TrajectoryPublisher(Node):

    def __init__(self):
        super().__init__('trajectory_publisher')
        self.declare_parameter('trajectory_file', 'baylands_nurbs_01.csv')

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.pub = self.create_publisher(Path, '/trajectory', qos)
        self.timer = self.create_timer(1.0, self.publish_path)

        pkg_path = get_package_share_directory('vehicle_controller')
        filename = self.get_parameter('trajectory_file').value
        csv_file = os.path.join(pkg_path, 'trajectories', filename)

        data   = np.loadtxt(csv_file, delimiter=",", skiprows=1)
        self.x = data[:, 0]
        self.y = data[:, 1]

        self.get_logger().info(
            f'Loaded {len(self.x)} control points from {filename}'
        )

    def publish_path(self):
        if self.pub.get_subscription_count() == 0:
            self.get_logger().info(
                'No subscribers to /trajectory yet. Waiting...',
                throttle_duration_sec=1.0
            )
            return

        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()

        for xi, yi in zip(self.x, self.y):
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = path.header.stamp
            pose.pose.position.x = float(xi)
            pose.pose.position.y = float(yi)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.pub.publish(path)
        self.get_logger().info(
            f'Published {len(path.poses)} control points'
        )
        self.timer.cancel()
        raise SystemExit


def main():
    rclpy.init()
    node = TrajectoryPublisher()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()