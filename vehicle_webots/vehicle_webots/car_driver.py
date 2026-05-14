"""ROS2 Tesla driver"""

import rclpy
import math
from geometry_msgs.msg import TwistStamped


WHEELBASE = 2.94      # metros — TeslaModel3
MAX_STEERING = 0.5    # rad — límite físico

class CarDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot

        self.__target_speed = 0.0
        self.__target_steer = 0.0

        rclpy.init(args=None)
        self.__node = rclpy.create_node('tesla_driver')
        self.__node.create_subscription(
            TwistStamped,
            '/cmd_vel',
            self.__cmd_vel_callback,
            1
        )

    def __cmd_vel_callback(self, msg):
        v = msg.twist.linear.x
        omega = msg.twist.angular.z  # = v/L * tan(delta)

        self.__target_speed = v

        # Conversión inversa: delta = atan(omega * L / v)
        if abs(v) > 0.01:
            steer = math.atan2(omega * WHEELBASE, abs(v))
        else:
            steer = 0.0

        self.__target_steer = max(-MAX_STEERING, min(MAX_STEERING, steer))

        self.__robot.setCruisingSpeed(self.__target_speed)
        self.__robot.setSteeringAngle(self.__target_steer)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)