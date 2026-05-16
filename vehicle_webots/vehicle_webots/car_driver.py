"""ROS2 Tesla driver - cmd_vel + joint_states + odometría Ackermann."""

import rclpy
import math
import numpy as np
from geometry_msgs.msg import TwistStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu, MagneticField, NavSatFix, NavSatStatus
from tf2_ros import TransformBroadcaster

# Parámetros Tesla Model 3
WHEELBASE    = 2.94
TRACK_REAR   = 1.72
TRACK_FRONT  = 1.72
WHEEL_RADIUS = 0.36
MAX_STEERING = 0.5

# Frecuencias sensores
IMU_PERIOD_MS = 10     # 100 Hz
MAG_PERIOD_MS = 20     # 50 Hz
GPS_PERIOD_MS = 100    # 10 Hz
CAM_PERIOD_MS = 33     # 30 Hz

# Ruido IMU
GYR_STDDEV     = 0.000864
GYR_BIAS_MEAN  = 0.000024
GYR_BIAS_STD   = 0.000005
ACC_STDDEV     = 0.00194
ACC_BIAS_MEAN  = 0.0004
ACC_BIAS_STD   = 0.0001

# Ruido GPS 
GPS_STDDEV     = 0.00002

# Ruido Magnetómetro 
MAG_STDDEV     = 8e-8       # Tesla


class CarDriver:
    def init(self, webots_node, properties):
        self.__robot    = webots_node.robot
        self.__timestep = int(self.__robot.getBasicTimeStep())

        self.__self_node = self.__robot.getSelf()

        # Sensores de posición de ruedas
        self.__left_rear_sensor   = self.__robot.getDevice('left_rear_sensor')
        self.__right_rear_sensor  = self.__robot.getDevice('right_rear_sensor')
        self.__left_steer_sensor  = self.__robot.getDevice('left_steer_sensor')
        self.__right_steer_sensor = self.__robot.getDevice('right_steer_sensor')

        self.__left_rear_sensor.enable(self.__timestep)
        self.__right_rear_sensor.enable(self.__timestep)
        self.__left_steer_sensor.enable(self.__timestep)
        self.__right_steer_sensor.enable(self.__timestep)

        # Sensores
        self.__imu        = self.__robot.getDevice('imu')
        self.__gyro       = self.__robot.getDevice('gyro')
        self.__accel      = self.__robot.getDevice('accel')
        self.__mag        = self.__robot.getDevice('mag')
        self.__gps        = self.__robot.getDevice('gps')

        # IMU stack
        self.__imu.enable(IMU_PERIOD_MS)
        self.__gyro.enable(IMU_PERIOD_MS)
        self.__accel.enable(IMU_PERIOD_MS)

        # Magnetómetro
        self.__mag.enable(MAG_PERIOD_MS)

        # GPS
        self.__gps.enable(GPS_PERIOD_MS)

        # ── Bias IMU (constante por sesión, simula sensor real) ───
        self.__gyro_bias  = np.random.normal(GYR_BIAS_MEAN, GYR_BIAS_STD, 3)
        self.__accel_bias = np.random.normal(ACC_BIAS_MEAN, ACC_BIAS_STD, 3)

        self.__last_imu_pub = 0.0
        self.__last_mag_pub = 0.0
        self.__last_gps_pub = 0.0

        # Estado odometría
        self.__x         = 0.0
        self.__y         = 0.0
        self.__theta     = 0.0
        self.__last_time = self.__robot.getTime()

        # Posiciones anteriores ruedas traseras (para calcular velocidad)
        self.__last_left_rear_pos  = 0.0
        self.__last_right_rear_pos = 0.0

        self.__target_speed = 0.0
        self.__target_steer = 0.0

        rclpy.init(args=None)
        self.__node = rclpy.create_node('tesla_driver')

        self.__node.create_subscription(
            TwistStamped, '/cmd_vel', self.__cmd_vel_callback, 1)

        self.__odom_pub    = self.__node.create_publisher(Odometry,      '/odom',              10)
        self.__imu_pub     = self.__node.create_publisher(Imu,           '/imu/data_raw',      10)
        self.__mag_pub     = self.__node.create_publisher(MagneticField, '/magnetometer',      10)
        self.__gps_pub     = self.__node.create_publisher(NavSatFix,     '/gps/fix',           10)
        self.__gt_pub      = self.__node.create_publisher(Odometry,      '/ground_truth_odom', 10)
        self.__joint_state_pub = self.__node.create_publisher(JointState, '/joint_states', 10)
        self.__tf_broadcaster  = TransformBroadcaster(self.__node)

    # Control
    def __cmd_vel_callback(self, msg):
        v     = msg.twist.linear.x
        omega = msg.twist.angular.z

        self.__target_speed = v

        # Inverse kinematics: phi = atan(l * omega / v)
        if abs(v) > 0.01:
            phi = -math.atan2(WHEELBASE * omega, v)
            if v < 0:
                phi = -phi
        else:
            phi = 0.0

        self.__target_steer = max(-MAX_STEERING, min(MAX_STEERING, phi))
        self.__robot.setCruisingSpeed(self.__target_speed)
        self.__robot.setSteeringAngle(self.__target_steer)

    # IMU: 
    def __publish_imu(self):
        now = self.__node.get_clock().now().to_msg()

        # Orientación (InertialUnit → quaternion)
        rpy   = self.__imu.getRollPitchYaw()
        r, p, y = rpy[0], rpy[1], rpy[2]
        cy, sy = math.cos(y*0.5), math.sin(y*0.5)
        cp, sp = math.cos(p*0.5), math.sin(p*0.5)
        cr, sr = math.cos(r*0.5), math.sin(r*0.5)

        msg = Imu()
        msg.header.stamp    = now
        msg.header.frame_id = 'base_link'

        msg.orientation.w = cr*cp*cy + sr*sp*sy
        msg.orientation.x = sr*cp*cy - cr*sp*sy
        msg.orientation.y = cr*sp*cy + sr*cp*sy
        msg.orientation.z = cr*cp*sy - sr*sp*cy
        # InertialUnit es ground truth — covarianza baja
        msg.orientation_covariance = [1e-6, 0, 0,
                                      0, 1e-6, 0,
                                      0, 0, 1e-6]

        # Velocidad angular con ruido + bias
        gyr = np.array(self.__gyro.getValues())
        gyr_noisy = gyr + self.__gyro_bias + np.random.normal(0, GYR_STDDEV, 3)
        msg.angular_velocity.x = gyr_noisy[0]
        msg.angular_velocity.y = gyr_noisy[1]
        msg.angular_velocity.z = gyr_noisy[2]
        s = GYR_STDDEV**2
        msg.angular_velocity_covariance = [s, 0, 0,
                                           0, s, 0,
                                           0, 0, s]

        # Aceleración lineal con ruido + bias
        acc = np.array(self.__accel.getValues())
        acc_noisy = acc + self.__accel_bias + np.random.normal(0, ACC_STDDEV, 3)
        msg.linear_acceleration.x = acc_noisy[0]
        msg.linear_acceleration.y = acc_noisy[1]
        msg.linear_acceleration.z = acc_noisy[2]
        s = ACC_STDDEV**2
        msg.linear_acceleration_covariance = [s, 0, 0,
                                              0, s, 0,
                                              0, 0, s]
        self.__imu_pub.publish(msg)

    # Compass: sensor_msgs/MagneticField
    def __publish_mag(self):
        now  = self.__node.get_clock().now().to_msg()
        vals = np.array(self.__mag.getValues())
        noisy = vals + np.random.normal(MAG_STDDEV, MAG_STDDEV, 3)

        msg = MagneticField()
        msg.header.stamp       = now
        msg.header.frame_id    = 'base_link'
        msg.magnetic_field.x   = noisy[0]
        msg.magnetic_field.y   = noisy[1]
        msg.magnetic_field.z   = noisy[2]
        s = MAG_STDDEV**2
        msg.magnetic_field_covariance = [s, 0, 0,
                                         0, s, 0,
                                         0, 0, s]
        self.__mag_pub.publish(msg)

    # GPS: sensor_msgs/NavSatFix
    def __publish_gps(self):
        now  = self.__node.get_clock().now().to_msg()

        vals = self.__gps.getValues()   #  [lat, lon, alt]

        msg = NavSatFix()
        msg.header.stamp         = now
        msg.header.frame_id      = 'gps_link'
        msg.status.status        = NavSatStatus.STATUS_FIX
        msg.status.service       = NavSatStatus.SERVICE_GPS
        msg.latitude = vals[0]
        msg.longitude = vals[1]
        msg.altitude = vals[2]

        msg.position_covariance  = [
            GPS_STDDEV**2, 0, 0,
            0, GPS_STDDEV**2, 0,
            0, 0, GPS_STDDEV**2
        ]
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.__gps_pub.publish(msg)

    # Ground truth odom
    def __publish_ground_truth(self):
        now = self.__node.get_clock().now().to_msg()

        # Pose exacta del simulador
        pos = self.__self_node.getPosition()
        rot = self.__self_node.getOrientation()

        # rot es matriz 3x3 flatten:
        # [r00 r01 r02 r10 r11 r12 r20 r21 r22]

        yaw = math.atan2(rot[3], rot[0])

        # Velocidades exactas
        vel = self.__self_node.getVelocity()
        # vel = [vx vy vz wx wy wz]

        msg = Odometry()
        msg.header.stamp = now
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = pos[0]
        msg.pose.pose.position.y = pos[1]
        msg.pose.pose.position.z = pos[2]

        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        # Velocidad global (ENU)
        vx_w = vel[0]
        vy_w = vel[1]
        vz_w = vel[2]

        # Transformar world -> base_link
        c = math.cos(yaw)
        s = math.sin(yaw)

        vx_b =  c * vx_w + s * vy_w
        vy_b = -s * vx_w + c * vy_w

        msg.twist.twist.linear.x = vx_b
        msg.twist.twist.linear.y = vy_b
        msg.twist.twist.linear.z = vz_w

        msg.twist.twist.angular.x = vel[3]
        msg.twist.twist.angular.y = vel[4]
        msg.twist.twist.angular.z = vel[5]

        self.__gt_pub.publish(msg)

    def __publish_joint_states(self, dt):
        now = self.__node.get_clock().now().to_msg()

        left_rear_pos  = self.__left_rear_sensor.getValue()
        right_rear_pos = self.__right_rear_sensor.getValue()
        phi_left       = self.__left_steer_sensor.getValue()
        phi_right      = self.__right_steer_sensor.getValue()

        # Velocidad angular ruedas traseras por diferencia de posición
        if dt > 0:
            left_rear_vel  = (left_rear_pos  - self.__last_left_rear_pos)  / dt
            right_rear_vel = (right_rear_pos - self.__last_right_rear_pos) / dt
        else:
            left_rear_vel  = 0.0
            right_rear_vel = 0.0

        self.__last_left_rear_pos  = left_rear_pos
        self.__last_right_rear_pos = right_rear_pos

        msg = JointState()
        msg.header.stamp = now
        msg.name     = ['left_rear_wheel', 'right_rear_wheel',
                        'left_steer',      'right_steer']
        msg.position = [left_rear_pos,  right_rear_pos,
                        phi_left,        phi_right]
        msg.velocity = [left_rear_vel,  right_rear_vel, 0.0, 0.0]
        self.__joint_state_pub.publish(msg)

        return left_rear_vel, right_rear_vel, phi_left, phi_right

    def __update_odometry(self, left_rear_vel, right_rear_vel, phi_left, phi_right, dt):
        if dt <= 0.0:
            return

        # ── Velocidad lineal del vehículo (Double Traction Axle odometry) ──
        # ── Reconstrucción del ángulo virtual de Ackermann ──
        if abs(phi_left) > 1e-6:
            phi_from_left = math.atan2(
                WHEELBASE * math.tan(phi_left),
                WHEELBASE + (TRACK_FRONT / 2.0) * math.tan(phi_left)
            )
        else:
            phi_from_left = 0.0

        if abs(phi_right) > 1e-6:
            phi_from_right = math.atan2(
                WHEELBASE * math.tan(phi_right),
                WHEELBASE - (TRACK_FRONT / 2.0) * math.tan(phi_right)
            )
        else:
            phi_from_right = 0.0

        # Ángulo virtual del eje delantero
        phi = -0.5 * (phi_from_left + phi_from_right)

        # ── Velocidad longitudinal desde encoders traseros ──
        v_left_rear  = left_rear_vel  * WHEEL_RADIUS
        v_right_rear = right_rear_vel * WHEEL_RADIUS

        # Radio de giro del centro del eje trasero
        if abs(phi) > 1e-6:
            R_b = WHEELBASE / math.tan(phi)

            R_left  = R_b - TRACK_REAR / 2.0
            R_right = R_b + TRACK_REAR / 2.0

            # Reconstrucción robusta desde ambas ruedas
            v_bx = 0.5 * (
                (v_left_rear  * (R_b / R_left)  if abs(R_left)  > 1e-6 else 0.0)
                +
                (v_right_rear * (R_b / R_right) if abs(R_right) > 1e-6 else 0.0)
            )

            omega_bz = (v_bx / WHEELBASE) * math.tan(phi)

        else:
            # Movimiento rectilíneo
            v_bx = 0.5 * (v_left_rear + v_right_rear)
            omega_bz = 0.0

        # ── Integración ──
        # Forward kinematics Ackermann:
        # x_dot     = v_bx * cos(theta)
        # y_dot     = v_bx * sin(theta)
        # theta_dot = v_bx / l * tan(phi)
        dtheta = omega_bz * dt
        dx     = v_bx * math.cos(self.__theta + dtheta / 2.0) * dt
        dy     = v_bx * math.sin(self.__theta + dtheta / 2.0) * dt

        self.__x     += dx
        self.__y     += dy
        self.__theta += dtheta

        now = self.__node.get_clock().now().to_msg()

        # TF odom → base_link
        tf = TransformStamped()
        tf.header.stamp    = now
        tf.header.frame_id = 'odom'
        tf.child_frame_id  = 'base_link'
        tf.transform.translation.x = self.__x
        tf.transform.translation.y = self.__y
        tf.transform.translation.z = 0.0
        tf.transform.rotation.z    = math.sin(self.__theta / 2.0)
        tf.transform.rotation.w    = math.cos(self.__theta / 2.0)
        self.__tf_broadcaster.sendTransform(tf)

        # Odometry msg
        odom = Odometry()
        odom.header.stamp       = now
        odom.header.frame_id    = 'odom'
        odom.child_frame_id     = 'base_link'
        odom.pose.pose.position.x    = self.__x
        odom.pose.pose.position.y    = self.__y
        odom.pose.pose.orientation.z = math.sin(self.__theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.__theta / 2.0)
        odom.twist.twist.linear.x    = v_bx
        odom.twist.twist.angular.z   = omega_bz
        self.__odom_pub.publish(odom)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)

        current_time = self.__robot.getTime()
        dt = current_time - self.__last_time
        self.__last_time = current_time

        left_vel, right_vel, phi_l, phi_r = self.__publish_joint_states(dt)
        self.__update_odometry(left_vel, right_vel, phi_l, phi_r, dt)
        if current_time - self.__last_imu_pub >= 0.005:
            self.__publish_imu()
            self.__last_imu_pub = current_time

        if current_time - self.__last_mag_pub >= 0.02:
            self.__publish_mag()
            self.__last_mag_pub = current_time

        if current_time - self.__last_gps_pub >= 0.1:
            self.__publish_gps()
            self.__last_gps_pub = current_time
        self.__publish_ground_truth()