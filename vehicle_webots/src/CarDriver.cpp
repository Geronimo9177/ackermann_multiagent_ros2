#include "vehicle_webots/CarDriver.hpp"

#include <cmath>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"

// ── Vehicle constants ─────────────────────────────────────────────
static constexpr double WHEELBASE    = 2.94;
static constexpr double TRACK_REAR   = 1.72;
static constexpr double TRACK_FRONT  = 1.72;
static constexpr double WHEEL_RADIUS = 0.36;
static constexpr double MAX_STEERING = 0.5;

// ── Sensor periods (seconds) ──────────────────────────────────────
static constexpr double IMU_PERIOD = 0.01;    // 100 Hz
static constexpr double MAG_PERIOD = 0.02;    // 50 Hz
static constexpr double GPS_PERIOD = 0.1;     // 10 Hz

// ── IMU noise ─────────────────────────────────────────────────────
static constexpr double GYR_STDDEV    = 0.000864;
static constexpr double GYR_BIAS_MEAN = 0.000024;
static constexpr double GYR_BIAS_STD  = 0.000005;
static constexpr double ACC_STDDEV    = 0.00194;
static constexpr double ACC_BIAS_MEAN = 0.0004;
static constexpr double ACC_BIAS_STD  = 0.0001;

// ── GPS / Mag noise ───────────────────────────────────────────────
static constexpr double GPS_STDDEV = 1.5;
static constexpr double MAG_STDDEV = 8e-8;

namespace vehicle_webots {

// ─────────────────────────────────────────────────────────────────
void CarDriver::init(webots_ros2_driver::WebotsNode *node,
                     std::unordered_map<std::string, std::string> &)
{
    node_ = node;
    int timestep = static_cast<int>(wb_robot_get_basic_time_step());

    // ── Wheel sensors ─────────────────────────────────────────────
    left_rear_sensor_   = wb_robot_get_device("left_rear_sensor");
    right_rear_sensor_  = wb_robot_get_device("right_rear_sensor");
    left_steer_sensor_  = wb_robot_get_device("left_steer_sensor");
    right_steer_sensor_ = wb_robot_get_device("right_steer_sensor");

    wb_position_sensor_enable(left_rear_sensor_,   timestep);
    wb_position_sensor_enable(right_rear_sensor_,  timestep);
    wb_position_sensor_enable(left_steer_sensor_,  timestep);
    wb_position_sensor_enable(right_steer_sensor_, timestep);

    // ── IMU stack ─────────────────────────────────────────────────
    imu_   = wb_robot_get_device("imu");
    gyro_  = wb_robot_get_device("gyro");
    accel_ = wb_robot_get_device("accel");
    mag_   = wb_robot_get_device("mag");
    gps_   = wb_robot_get_device("gps");

    wb_inertial_unit_enable(imu_,   static_cast<int>(IMU_PERIOD * 1000));
    wb_gyro_enable(gyro_,           static_cast<int>(IMU_PERIOD * 1000));
    wb_accelerometer_enable(accel_, static_cast<int>(IMU_PERIOD * 1000));
    wb_compass_enable(mag_,         static_cast<int>(MAG_PERIOD * 1000));
    wb_gps_enable(gps_,             static_cast<int>(GPS_PERIOD * 1000));

    self_node_ = wb_supervisor_node_get_self();

    // ── RNG & bias ────────────────────────────────────────────────
    rng_.seed(std::random_device{}());
    std::normal_distribution<double> bias_gyr(GYR_BIAS_MEAN, GYR_BIAS_STD);
    std::normal_distribution<double> bias_acc(ACC_BIAS_MEAN, ACC_BIAS_STD);
    for (int i = 0; i < 3; ++i) {
        gyro_bias_[i]  = bias_gyr(rng_);
        accel_bias_[i] = bias_acc(rng_);
    }
    gyr_noise_ = std::normal_distribution<double>(0.0, GYR_STDDEV);
    acc_noise_ = std::normal_distribution<double>(0.0, ACC_STDDEV);
    mag_noise_ = std::normal_distribution<double>(0.0, MAG_STDDEV);

    // ── Publishers ────────────────────────────────────────────────
    odom_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
    gt_pub_   = node->create_publisher<nav_msgs::msg::Odometry>("/ground_truth_odom", 10);
    imu_pub_  = node->create_publisher<sensor_msgs::msg::Imu>("/imu/data_raw", 10);
    mag_pub_  = node->create_publisher<sensor_msgs::msg::MagneticField>("/magnetometer", 10);
    gps_pub_  = node->create_publisher<sensor_msgs::msg::NavSatFix>("/gps/fix", 10);
    js_pub_   = node->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node);

    // ── Subscriber ────────────────────────────────────────────────
    cmd_vel_sub_ = node->create_subscription<geometry_msgs::msg::TwistStamped>(
        "/cmd_vel", 1,
        std::bind(&CarDriver::cmdVelCallback, this, std::placeholders::_1));

    last_time_ = wb_robot_get_time();

    RCLCPP_INFO(node->get_logger(), "CarDriver C++ plugin initialized.");
}

// ─────────────────────────────────────────────────────────────────
void CarDriver::cmdVelCallback(
  const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
    double v     = msg->twist.linear.x;
    double omega = msg->twist.angular.z;

    target_speed_ = v;

    double phi = 0.0;
    if (std::abs(v) > 0.01) {
        phi = -std::atan2(WHEELBASE * omega, v);
        if (v < 0.0) phi = -phi;
    }

    target_steer_ = std::clamp(phi, -MAX_STEERING, MAX_STEERING);
    wbu_driver_set_cruising_speed(target_speed_);
    wbu_driver_set_steering_angle(target_steer_);
}

// ─────────────────────────────────────────────────────────────────
void CarDriver::publishImu()
{
  auto now = node_->get_clock()->now();

  // Orientation from InertialUnit
  const double *quat = wb_inertial_unit_get_quaternion(imu_);
  // Webots quaternion: [x, y, z, w]

  // Gyro with noise + bias
  const double *gyr = wb_gyro_get_values(gyro_);
  // Accel with noise + bias
  const double *acc = wb_accelerometer_get_values(accel_);

  sensor_msgs::msg::Imu msg;
  msg.header.stamp    = now;
  msg.header.frame_id = "base_link";

  msg.orientation.x = quat[0];
  msg.orientation.y = quat[1];
  msg.orientation.z = quat[2];
  msg.orientation.w = quat[3];
  msg.orientation_covariance = {1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1e-6};

  double s_gyr = GYR_STDDEV * GYR_STDDEV;
  msg.angular_velocity.x = gyr[0] + gyro_bias_[0] + gyr_noise_(rng_);
  msg.angular_velocity.y = gyr[1] + gyro_bias_[1] + gyr_noise_(rng_);
  msg.angular_velocity.z = gyr[2] + gyro_bias_[2] + gyr_noise_(rng_);
  msg.angular_velocity_covariance = {s_gyr, 0, 0, 0, s_gyr, 0, 0, 0, s_gyr};

  double s_acc = ACC_STDDEV * ACC_STDDEV;
  msg.linear_acceleration.x = acc[0] + accel_bias_[0] + acc_noise_(rng_);
  msg.linear_acceleration.y = acc[1] + accel_bias_[1] + acc_noise_(rng_);
  msg.linear_acceleration.z = acc[2] + accel_bias_[2] + acc_noise_(rng_);
  msg.linear_acceleration_covariance = {s_acc, 0, 0, 0, s_acc, 0, 0, 0, s_acc};

  imu_pub_->publish(msg);
}

// ─────────────────────────────────────────────────────────────────
void CarDriver::publishMag()
{
  auto now = node_->get_clock()->now();
  const double *vals = wb_compass_get_values(mag_);

  double s = MAG_STDDEV * MAG_STDDEV;
  sensor_msgs::msg::MagneticField msg;
  msg.header.stamp       = now;
  msg.header.frame_id    = "base_link";
  msg.magnetic_field.x   = vals[0] + mag_noise_(rng_);
  msg.magnetic_field.y   = vals[1] + mag_noise_(rng_);
  msg.magnetic_field.z   = vals[2] + mag_noise_(rng_);
  msg.magnetic_field_covariance = {s, 0, 0, 0, s, 0, 0, 0, s};

  mag_pub_->publish(msg);
}

// ─────────────────────────────────────────────────────────────────
void CarDriver::publishGps()
{
  auto now  = node_->get_clock()->now();
  const double *vals = wb_gps_get_values(gps_);  // [lat, lon, alt] WGS84

  double s = GPS_STDDEV * GPS_STDDEV;
  sensor_msgs::msg::NavSatFix msg;
  msg.header.stamp    = now;
  msg.header.frame_id = "gps_link";
  msg.status.status   = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
  msg.status.service  = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;
  msg.latitude        = vals[0];
  msg.longitude       = vals[1];
  msg.altitude        = vals[2];
  msg.position_covariance = {s, 0, 0, 0, s, 0, 0, 0, s};
  msg.position_covariance_type =
    sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN;

  gps_pub_->publish(msg);
}

// ─────────────────────────────────────────────────────────────────
void CarDriver::publishGroundTruth()
{
  auto now = node_->get_clock()->now();

  // Ground truth pose
  const double *pos = wb_supervisor_node_get_position(self_node_);

  // Rotation matrix 
  const double *rot = wb_supervisor_node_get_orientation(self_node_);
  // rot = [r00 r01 r02 r10 r11 r12 r20 r21 r22]
  // yaw = atan2(r10, r00)
  double yaw = std::atan2(rot[3], rot[0]);

  // Ground truth velocity in world frame
  const double *vel = wb_supervisor_node_get_velocity(self_node_);

  // Transform velocity to body frame
  double c   = std::cos(yaw);
  double s_y = std::sin(yaw);
  double vx_b =  c * vel[0] + s_y * vel[1];
  double vy_b = -s_y * vel[0] + c * vel[1];

  nav_msgs::msg::Odometry msg;
  msg.header.stamp         = now;
  msg.header.frame_id      = "map";
  msg.child_frame_id       = "base_link";

  msg.pose.pose.position.x = pos[0];
  msg.pose.pose.position.y = pos[1];
  msg.pose.pose.position.z = pos[2];
  msg.pose.pose.orientation.z = std::sin(yaw / 2.0);
  msg.pose.pose.orientation.w = std::cos(yaw / 2.0);

  msg.twist.twist.linear.x  = vx_b;
  msg.twist.twist.linear.y  = vy_b;
  msg.twist.twist.linear.z  = vel[2];
  msg.twist.twist.angular.x = vel[3];
  msg.twist.twist.angular.y = vel[4];
  msg.twist.twist.angular.z = vel[5];

  gt_pub_->publish(msg);
}

// ─────────────────────────────────────────────────────────────────
std::array<double, 4> CarDriver::publishJointStates(double dt)
{
  auto now = node_->get_clock()->now();

  double lp = wb_position_sensor_get_value(left_rear_sensor_);
  double rp = wb_position_sensor_get_value(right_rear_sensor_);
  double pl = wb_position_sensor_get_value(left_steer_sensor_);
  double pr = wb_position_sensor_get_value(right_steer_sensor_);

  double lv = (dt > 0) ? (lp - last_left_pos_)  / dt : 0.0;
  double rv = (dt > 0) ? (rp - last_right_pos_) / dt : 0.0;
  last_left_pos_  = lp;
  last_right_pos_ = rp;

  sensor_msgs::msg::JointState msg;
  msg.header.stamp = now;
  msg.name     = {"left_rear_wheel", "right_rear_wheel",
                  "left_steer",      "right_steer"};
  msg.position = {lp, rp, pl, pr};
  msg.velocity = {lv, rv, 0.0, 0.0};

  js_pub_->publish(msg);
  return {lv, rv, pl, pr};
}

// ─────────────────────────────────────────────────────────────────
void CarDriver::updateOdometry(double lv, double rv,
                               double phi_l, double phi_r, double dt)
{
  if (dt <= 0.0) return;

  // Reconstruct virtual Ackermann angle from both steer sensors
  auto phi_from = [](double phi, double sign) -> double {
    if (std::abs(phi) < 1e-6) return 0.0;
    return std::atan2(WHEELBASE * std::tan(phi),
                      WHEELBASE + sign * (TRACK_FRONT / 2.0) * std::tan(phi));
  };

  double phi = -0.5 * (phi_from(phi_l,  1.0) + phi_from(phi_r, -1.0));

  double v_l = lv * WHEEL_RADIUS;
  double v_r = rv * WHEEL_RADIUS;

  double v_bx, omega_bz;
  if (std::abs(phi) > 1e-6) {
    double R_b     = WHEELBASE / std::tan(phi);
    double R_left  = R_b - TRACK_REAR / 2.0;
    double R_right = R_b + TRACK_REAR / 2.0;
    double vl_c = (std::abs(R_left)  > 1e-6) ? v_l * R_b / R_left  : 0.0;
    double vr_c = (std::abs(R_right) > 1e-6) ? v_r * R_b / R_right : 0.0;
    v_bx    = 0.5 * (vl_c + vr_c);
    omega_bz = (v_bx / WHEELBASE) * std::tan(phi);
  } else {
    v_bx    = 0.5 * (v_l + v_r);
    omega_bz = 0.0;
  }

  double dtheta = omega_bz * dt;
  double mid    = theta_ + dtheta * 0.5;
  x_     += v_bx * std::cos(mid) * dt;
  y_     += v_bx * std::sin(mid) * dt;
  theta_ += dtheta;

  auto now = node_->get_clock()->now();

  // TF odom → base_link
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp    = now;
  tf.header.frame_id = "odom";
  tf.child_frame_id  = "base_link";
  tf.transform.translation.x = x_;
  tf.transform.translation.y = y_;
  tf.transform.translation.z = 0.0;
  tf.transform.rotation.z    = std::sin(theta_ / 2.0);
  tf.transform.rotation.w    = std::cos(theta_ / 2.0);
  tf_broadcaster_->sendTransform(tf);

  // Odometry message
  nav_msgs::msg::Odometry odom;
  odom.header.stamp       = now;
  odom.header.frame_id    = "odom";
  odom.child_frame_id     = "base_link";
  odom.pose.pose.position.x    = x_;
  odom.pose.pose.position.y    = y_;
  odom.pose.pose.orientation.z = std::sin(theta_ / 2.0);
  odom.pose.pose.orientation.w = std::cos(theta_ / 2.0);
  odom.twist.twist.linear.x    = v_bx;
  odom.twist.twist.angular.z   = omega_bz;

  // Covariance
  odom.pose.covariance[0]  = 0.001;  // x
  odom.pose.covariance[7]  = 0.001;  // y
  odom.pose.covariance[14] = 1.0;    // z
  odom.pose.covariance[21] = 1.0;    // roll
  odom.pose.covariance[28] = 1.0;    // pitch
  odom.pose.covariance[35] = 0.005;  // yaw

  odom.twist.covariance[0]  = 0.001; // vx
  odom.twist.covariance[7]  = 1.0;   // vy
  odom.twist.covariance[14] = 1.0;   // vz
  odom.twist.covariance[21] = 1.0;   // vroll
  odom.twist.covariance[28] = 1.0;   // vpitch
  odom.twist.covariance[35] = 0.02;  // vyaw

  odom_pub_->publish(odom);
}

// ─────────────────────────────────────────────────────────────────
void CarDriver::step()
{
  double current_time = wb_robot_get_time();
  double dt = current_time - last_time_;
  last_time_ = current_time;

  auto [lv, rv, phi_l, phi_r] = publishJointStates(dt);
  updateOdometry(lv, rv, phi_l, phi_r, dt);

  if (current_time - last_imu_pub_ >= IMU_PERIOD) {
    publishImu();
    last_imu_pub_ = current_time;
  }
  if (current_time - last_mag_pub_ >= MAG_PERIOD) {
    publishMag();
    last_mag_pub_ = current_time;
  }
  if (current_time - last_gps_pub_ >= GPS_PERIOD) {
    publishGps();
    last_gps_pub_ = current_time;
  }

  publishGroundTruth();
}

}  // namespace vehicle_webots

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(vehicle_webots::CarDriver,
                       webots_ros2_driver::PluginInterface)