#!/usr/bin/env python3
import copy

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class OdomCovarianceRepublisher(Node):
    def __init__(self):
        super().__init__("odom_covariance_republisher")

        self.declare_parameter(
            "input_topic",
            "/ackermann_steering_controller/odometry_raw",
        )
        self.declare_parameter("output_topic", "/ackermann_steering_controller/odometry")
        self.declare_parameter(
            "pose_covariance_diagonal",
            [0.0, 7.0, 14.0, 21.0, 28.0, 35.0],
        )

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        diag = self.get_parameter("pose_covariance_diagonal").value

        if len(diag) != 6:
            self.get_logger().warn(
                "pose_covariance_diagonal must have 6 values; using defaults"
            )
            diag = [0.0, 7.0, 14.0, 21.0, 28.0, 35.0]

        self.pose_covariance = self._diag_to_covariance(diag)

        self.sub = self.create_subscription(
            Odometry, self.input_topic, self.odom_callback, 10
        )
        self.pub = self.create_publisher(Odometry, self.output_topic, 10)

        self.get_logger().info(
            f"Republishing {self.input_topic} -> {self.output_topic} with pose covariance"
        )

    def _diag_to_covariance(self, diag):
        cov = [0.0] * 36
        for idx, val in enumerate(diag):
            cov[idx * 6 + idx] = float(val)
        return cov

    def odom_callback(self, msg):
        out_msg = copy.deepcopy(msg)
        out_msg.pose.covariance = self.pose_covariance
        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = OdomCovarianceRepublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
