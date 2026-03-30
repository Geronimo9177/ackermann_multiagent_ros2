#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import MagneticField

class MagRepublisher(Node):
    def __init__(self):
        super().__init__('mag_republisher')
        self.sub = self.create_subscription(
            MagneticField, '/magnetometer', self.cb, 10)
        self.pub = self.create_publisher(
            MagneticField, '/magnetometer/corrected', 10)

    def cb(self, msg: MagneticField):
        out = MagneticField()
        out.header = msg.header

        # Conversión de Gauss a Teslas
        # out.magnetic_field.x = msg.magnetic_field.x / 10000.0
        # out.magnetic_field.y = msg.magnetic_field.y / 10000.0
        # out.magnetic_field.z = msg.magnetic_field.z / 10000.0

        out.magnetic_field.x = msg.magnetic_field.y / 10000.0
        out.magnetic_field.y = msg.magnetic_field.x / 10000.0
        out.magnetic_field.z = - msg.magnetic_field.z / 10000.0

        # Covarianza realista para simulación
        var = (4e-7) ** 2
        out.magnetic_field_covariance = [
            var,  0.0,  0.0,
            0.0,  var,  0.0,
            0.0,  0.0,  var
        ]
        self.pub.publish(out)

def main():
    rclpy.init()
    rclpy.spin(MagRepublisher())

if __name__ == '__main__':
    main()