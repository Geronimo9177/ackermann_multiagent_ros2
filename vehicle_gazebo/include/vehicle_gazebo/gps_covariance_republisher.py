#!/usr/bin/env python3
"""Republish GPS NavSatFix with configurable non-zero position covariance."""

import copy

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


class GpsCovarianceRepublisher(Node):
    def __init__(self):
        super().__init__("gps_covariance_republisher")

        self.declare_parameter("input_topic", "/gps/fix_raw")
        self.declare_parameter("output_topic", "/gps/fix")

        # Covariance in ENU meters^2 for NavSatFix.position_covariance.
        self.declare_parameter(
            "position_covariance_diagonal",
            [1.5**2, 1.5**2, 3.0**2],
        )

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        diag = self.get_parameter("position_covariance_diagonal").value

        self.position_covariance = self._diag_to_covariance(
            diag, "position_covariance_diagonal"
        )

        self.sub = self.create_subscription(NavSatFix, self.input_topic, self.gps_callback, 20)
        self.pub = self.create_publisher(NavSatFix, self.output_topic, 20)

        self.get_logger().info(
            f"Republishing {self.input_topic} -> {self.output_topic} with GPS covariance "
            f"diag=[{self.position_covariance[0]:.4f}, {self.position_covariance[4]:.4f}, {self.position_covariance[8]:.4f}]"
        )

    def _diag_to_covariance(self, diag, param_name):
        if len(diag) != 3:
            self.get_logger().warn(
                f"{param_name} must have 3 values; using defaults"
            )
            diag = [1.5**2, 1.5**2, 3.0**2]

        cov = [0.0] * 9
        for idx, val in enumerate(diag):
            cov[idx * 3 + idx] = float(val)
        return cov

    def gps_callback(self, msg: NavSatFix):
        out_msg = copy.deepcopy(msg)
        out_msg.position_covariance = self.position_covariance
        out_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = GpsCovarianceRepublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()