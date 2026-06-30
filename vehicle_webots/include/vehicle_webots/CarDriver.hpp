#ifndef VEHICLE_WEBOTS_CAR_DRIVER_HPP
#define VEHICLE_WEBOTS_CAR_DRIVER_HPP

#include <unordered_map>
#include <string>
#include <random>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "webots_ros2_driver/PluginInterface.hpp"
#include "webots_ros2_driver/WebotsNode.hpp"

#include <webots/inertial_unit.h>
#include <webots/gyro.h>
#include <webots/accelerometer.h>
#include <webots/compass.h>
#include <webots/gps.h>
#include <webots/position_sensor.h>
#include <webots/camera.h>
#include <webots/robot.h>
#include <webots/vehicle/driver.h>
#include <webots/supervisor.h>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include "tf2_ros/transform_broadcaster.h"

#include "std_msgs/msg/header.hpp"
#include "std_msgs/msg/bool.hpp"

namespace vehicle_webots {

class CarDriver : public webots_ros2_driver::PluginInterface {
public:
  void init(webots_ros2_driver::WebotsNode *node,
            std::unordered_map<std::string, std::string> &parameters) override;
  void step() override;

private:
  // ── Webots devices ───────────────────────────────────────────
  WbDeviceTag left_rear_sensor_;
  WbDeviceTag right_rear_sensor_;
  WbDeviceTag left_steer_sensor_;
  WbDeviceTag right_steer_sensor_;
  WbDeviceTag imu_;
  WbDeviceTag gyro_;
  WbDeviceTag accel_;
  WbDeviceTag mag_;
  WbDeviceTag gps_;
  WbDeviceTag camera_;

  WbNodeRef self_node_;

  // ── ROS publishers ───────────────────────────────────────────
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr                       odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr                       gt_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr                         imu_pub_;
  rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr               mag_pub_;
  rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr                   gps_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr                       seg_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr                  js_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr set_pose_local_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr set_pose_global_pub_;

  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  webots_ros2_driver::WebotsNode *node_;

  // ── State ────────────────────────────────────────────────────
  double x_{0.0}, y_{0.0}, theta_{0.0};
  double last_time_{0.0};
  double last_left_pos_{0.0}, last_right_pos_{0.0};
  double target_speed_{0.0}, target_steer_{0.0};

  WbFieldRef trans_field_;
  WbFieldRef rot_field_;
  WbNodeRef  viewpoint_node_;

  // ── Sensor timing ────────────────────────────────────────────
  double last_imu_pub_{0.0};
  double last_mag_pub_{0.0};
  double last_gps_pub_{0.0};
  double last_cam_pub_{0.0};

  // ── Camera info ───────────────────────────────────────────────
  int cam_width_{0};
  int cam_height_{0};
  int cam_pixels_{0};  // cam_width_ * cam_height_

  // ── IMU bias (initialized once) ──────────────────────────────
  std::array<double, 3> gyro_bias_{};
  std::array<double, 3> accel_bias_{};

  // ── RNG ──────────────────────────────────────────────────────
  std::mt19937 rng_;
  std::normal_distribution<double> gyr_noise_;
  std::normal_distribution<double> acc_noise_;
  std::normal_distribution<double> mag_noise_;

  // ── RL sync ──────────────────────────────────────────────────
  static constexpr double RL_PERIOD = 0.05;   // 50 ms sim time

  // Two-phase state machine:
  //   RUNNING    → sim runs FAST until RL_PERIOD elapses
  //   WAIT_MPC   → trigger sent, sim in REAL_TIME, waiting for MPC /cmd_vel_mpc
  //   WAIT_PPO   → MPC cmd received, sim PAUSED, waiting for PPO /cmd_vel
  enum class SyncState { RUNNING, WAIT_MPC, WAIT_PPO };

  bool      system_ready_{false};
  bool      training_mode_{false};
  SyncState sync_state_{SyncState::RUNNING};
  double    last_rl_trigger_{-1.0};

  // Timestamps to detect *new* commands arriving after our trigger
  int64_t trigger_stamp_ns_{0};
  int64_t last_mpc_stamp_ns_{0};   // stamp of last /cmd_vel_mpc seen
  int64_t last_ppo_stamp_ns_{0};   // stamp of last /cmd_vel seen

  // MPC command stored as fallback — applied only if PPO times out
  double mpc_fallback_v_{0.0};
  double mpc_fallback_steer_{0.0};

  // Safety timeouts (wall-clock milliseconds)
  static constexpr int MPC_TIMEOUT_MS = 300;   // max wait for MPC
  static constexpr int PPO_TIMEOUT_MS = 3000;   // max wait for PPO (training)

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr   start_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr   reset_sub_;

  // Intermediate MPC command (arrives on /cmd_vel_mpc)
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr mpc_cmd_sub_;

  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr    rl_trigger_pub_;

  // ── Methods ──────────────────────────────────────────────────
  void cmdVelCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
  void mpcCmdCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
  void publishImu();
  void publishMag();
  void publishGps();
  void publishCamera();
  void publishGroundTruth();
  std::array<double, 4> publishJointStates(double dt);
  void updateOdometry(double lv, double rv,
                      double phi_l, double phi_r, double dt);

  // Spin-wait helper: spins ROS callbacks until predicate() is true or
  // wall-clock deadline is exceeded.  Returns true if predicate satisfied.
  template<typename Pred>
  bool spinUntil(Pred predicate, int timeout_ms);
};

}  // namespace vehicle_webots
#endif