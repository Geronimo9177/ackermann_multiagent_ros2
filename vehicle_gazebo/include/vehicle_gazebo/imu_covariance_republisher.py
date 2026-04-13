#!/usr/bin/env python3
"""Republish IMU with configurable covariance values for downstream filters."""

import copy
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuCovarianceRepublisher(Node):
    def __init__(self):
        super().__init__("imu_covariance_republisher")

        self.declare_parameter("input_topic", "/imu/data")
        self.declare_parameter("output_topic", "/imu/data_cov")

        # Orientation covariance is not published by imu_complementary_filter.
        # We inject practical values for robot_localization.
        self.declare_parameter(
            "orientation_covariance_diagonal",
            [(0.01 / 9.81) ** 2, (0.01 / 9.81) ** 2, math.radians(5.0) ** 2],
        )

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)

        orientation_diag = self.get_parameter(
            "orientation_covariance_diagonal"
        ).value

        self.orientation_covariance = self._diag_to_covariance(
            orientation_diag, "orientation_covariance_diagonal"
        )

        self.sub = self.create_subscription(Imu, self.input_topic, self.imu_callback, 20)
        self.pub = self.create_publisher(Imu, self.output_topic, 20)

        self.get_logger().info(
            f"Republishing {self.input_topic} -> {self.output_topic} with tuned IMU covariances"
        )

    def _diag_to_covariance(self, diag, param_name):
        if len(diag) != 3:
            self.get_logger().warn(
                f"{param_name} must have 3 values; using identity defaults"
            )
            diag = [1.0, 1.0, 1.0]

        cov = [0.0] * 9
        for idx, val in enumerate(diag):
            cov[idx * 3 + idx] = float(val)
        return cov

    def imu_callback(self, msg):
        out_msg = copy.deepcopy(msg)
        out_msg.orientation_covariance = self.orientation_covariance
        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = ImuCovarianceRepublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()