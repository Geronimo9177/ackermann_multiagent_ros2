#include "vehicle_webots/CarDriver.hpp"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include <tf2/LinearMath/Matrix3x3.h>

#include "std_msgs/msg/header.hpp"
#include "std_msgs/msg/bool.hpp"

// ── Vehicle constants ─────────────────────────────────────────────
static constexpr double WHEELBASE    = 2.94;
static constexpr double TRACK_REAR   = 1.72;
static constexpr double TRACK_FRONT  = 1.72;
static constexpr double WHEEL_RADIUS = 0.36;
static constexpr double MAX_STEERING = 0.5;

// ── Sensor periods (seconds) ──────────────────────────────────────
static constexpr double IMU_PERIOD = 0.01;
static constexpr double MAG_PERIOD = 0.02;
static constexpr double GPS_PERIOD = 0.1;
static constexpr double CAM_PERIOD = 0.05;

// ── IMU noise ─────────────────────────────────────────────────────
static constexpr double GYR_STDDEV    = 0.000864;
static constexpr double GYR_BIAS_MEAN = 0.000024;
static constexpr double GYR_BIAS_STD  = 0.000005;
static constexpr double ACC_STDDEV    = 0.00194;
static constexpr double ACC_BIAS_MEAN = 0.0004;
static constexpr double ACC_BIAS_STD  = 0.0001;

static constexpr double GPS_STDDEV = 1.5;
static constexpr double MAG_STDDEV = 8e-8;

namespace vehicle_webots {

// ─────────────────────────────────────────────────────────────────
// Helper: spin ROS callbacks until predicate() or timeout
// ─────────────────────────────────────────────────────────────────
template<typename Pred>
bool CarDriver::spinUntil(Pred predicate, int timeout_ms)
{
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(timeout_ms);
    while (!predicate() && std::chrono::steady_clock::now() < deadline) {
        rclcpp::spin_some(node_->get_node_base_interface());
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return predicate();
}

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

    // ── Sensor stack ──────────────────────────────────────────────
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

    camera_ = wb_robot_get_device("camera");
    wb_camera_enable(camera_,             static_cast<int>(CAM_PERIOD * 1000));
    wb_camera_recognition_enable(camera_, static_cast<int>(CAM_PERIOD * 1000));
    wb_camera_recognition_enable_segmentation(camera_);

    self_node_   = wb_supervisor_node_get_self();
    trans_field_ = wb_supervisor_node_get_field(self_node_, "translation");
    rot_field_   = wb_supervisor_node_get_field(self_node_, "rotation");
    viewpoint_node_ = wb_supervisor_node_get_from_def("VIEWPOINT");

    cam_width_  = wb_camera_get_width(camera_);
    cam_height_ = wb_camera_get_height(camera_);
    cam_pixels_ = cam_width_ * cam_height_;

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

    // ── RL mode ───────────────────────────────────────────────────
    training_mode_ = node->declare_parameter("training_mode", false);

    // ── Publishers ────────────────────────────────────────────────
    odom_pub_            = node->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
    gt_pub_              = node->create_publisher<nav_msgs::msg::Odometry>("/ground_truth_odom", 10);
    imu_pub_             = node->create_publisher<sensor_msgs::msg::Imu>("/imu/data_raw", 10);
    mag_pub_             = node->create_publisher<sensor_msgs::msg::MagneticField>("/magnetometer", 10);
    gps_pub_             = node->create_publisher<sensor_msgs::msg::NavSatFix>("/gps/fix", 10);
    seg_pub_             = node->create_publisher<sensor_msgs::msg::Image>("/camera/segmentation", 10);
    js_pub_              = node->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    set_pose_local_pub_  = node->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/ekf_local/set_pose", 1);
    set_pose_global_pub_ = node->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/ekf_global/set_pose", 1);

    rl_trigger_pub_ = node->create_publisher<std_msgs::msg::Header>("/sim/trigger", 1);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node);

    // ── Subscribers ───────────────────────────────────────────────

    // /sim/start  — activates the RL loop
    start_sub_ = node->create_subscription<std_msgs::msg::Bool>(
        "/sim/start", 1,
        [this](const std_msgs::msg::Bool::SharedPtr msg) {
            if (msg->data && !system_ready_) {
                system_ready_ = true;
                sync_state_   = SyncState::RUNNING;
                last_rl_trigger_ = -1.0;
                RCLCPP_INFO(node_->get_logger(), "RL start → sync loop active");
                if (training_mode_)
                    wb_supervisor_simulation_set_mode(WB_SUPERVISOR_SIMULATION_MODE_FAST);
            }
        });

    // /sim/reset  — episode reset
    reset_sub_ = node->create_subscription<std_msgs::msg::Bool>(
        "/sim/reset", 1,
        [this](const std_msgs::msg::Bool::SharedPtr) {
            system_ready_ = false;
            sync_state_   = SyncState::RUNNING;
            last_rl_trigger_ = -1.0;
            last_mpc_stamp_ns_ = 0;
            last_ppo_stamp_ns_ = 0;
            target_speed_ = 0.0;
            target_steer_ = 0.0;
            wbu_driver_set_cruising_speed(0.0);
            wbu_driver_set_steering_angle(0.0);

            wb_supervisor_simulation_set_mode(WB_SUPERVISOR_SIMULATION_MODE_REAL_TIME);
            wb_supervisor_node_reset_physics(self_node_);

            const double origin_trans[3] = {0.0, 0.0, 0.0};
            const double origin_rot[4]   = {0.0, 0.0, 1.0, 0.0};
            wb_supervisor_field_set_sf_vec3f(trans_field_,  origin_trans);
            wb_supervisor_field_set_sf_rotation(rot_field_, origin_rot);

            if (viewpoint_node_) {
                WbFieldRef vp  = wb_supervisor_node_get_field(viewpoint_node_, "position");
                WbFieldRef vr  = wb_supervisor_node_get_field(viewpoint_node_, "orientation");
                const double cp[3] = {-20.0,  8.0, 6.0};
                const double cr[4] = {  0.0, 0.4, -0.8, 0.35};
                wb_supervisor_field_set_sf_vec3f(vp, cp);
                wb_supervisor_field_set_sf_rotation(vr, cr);
            }

            x_ = 0.0; y_ = 0.0; theta_ = 0.0;
            last_left_pos_  = wb_position_sensor_get_value(left_rear_sensor_);
            last_right_pos_ = wb_position_sensor_get_value(right_rear_sensor_);

            // Reset EKF pose
            geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
            pose_msg.header.stamp = node_->get_clock()->now();
            pose_msg.pose.pose.orientation.w = 1.0;
            pose_msg.pose.covariance.fill(0.0);
            pose_msg.pose.covariance[0]  = 1e-9;
            pose_msg.pose.covariance[7]  = 1e-9;
            pose_msg.pose.covariance[14] = 1e-9;
            pose_msg.pose.covariance[21] = 1e-9;
            pose_msg.pose.covariance[28] = 1e-9;
            pose_msg.pose.covariance[35] = 1e-9;
            pose_msg.header.frame_id = "odom";
            set_pose_local_pub_->publish(pose_msg);
            pose_msg.header.frame_id = "map";
            set_pose_global_pub_->publish(pose_msg);

            RCLCPP_INFO(node_->get_logger(), "Simulation reset");
        });

    // /cmd_vel_mpc  — intermediate MPC command (Phase 1 → Phase 2 transition)
    mpc_cmd_sub_ = node->create_subscription<geometry_msgs::msg::TwistStamped>(
        "/cmd_vel_mpc", 1,
        std::bind(&CarDriver::mpcCmdCallback, this, std::placeholders::_1));

    // /cmd_vel  — final command from PPO (Phase 2 → RUNNING transition)
    cmd_vel_sub_ = node->create_subscription<geometry_msgs::msg::TwistStamped>(
        "/cmd_vel", 1,
        std::bind(&CarDriver::cmdVelCallback, this, std::placeholders::_1));

    last_time_ = wb_robot_get_time();
    RCLCPP_INFO(node->get_logger(), "CarDriver C++ plugin initialized.");
}

// ─────────────────────────────────────────────────────────────────
// Phase 1 complete: MPC published its command
// → apply it to actuators, transition to WAIT_PPO, pause sim
// ─────────────────────────────────────────────────────────────────
void CarDriver::mpcCmdCallback(
    const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
    int64_t stamp = rclcpp::Time(msg->header.stamp).nanoseconds();
    last_mpc_stamp_ns_ = stamp;

    if (!system_ready_ || sync_state_ != SyncState::WAIT_MPC)
        return;

    // Only accept commands that were generated after our trigger
    if (stamp <= trigger_stamp_ns_)
        return;

    // Apply command to actuators
    double v     = msg->twist.linear.x;
    double omega = msg->twist.angular.z;
    double phi   = 0.0;
    if (std::abs(v) > 0.01) {
        phi = -std::atan2(WHEELBASE * omega, v);
        if (v < 0.0) phi = -phi;
    }
    target_steer_ = std::clamp(phi, -MAX_STEERING, MAX_STEERING);
    target_speed_ = v;
    wbu_driver_set_cruising_speed(v * 3.6);
    wbu_driver_set_steering_angle(target_steer_);

    // Transition: MPC done → pause and wait for PPO
    sync_state_ = SyncState::WAIT_PPO;

    if (training_mode_) {
        wb_supervisor_simulation_set_mode(WB_SUPERVISOR_SIMULATION_MODE_PAUSE);
        RCLCPP_DEBUG(node_->get_logger(),
            "MPC cmd received → PAUSE (waiting for PPO)");
    }
}

// ─────────────────────────────────────────────────────────────────
// Phase 2 complete: PPO published the final command
// → apply it, resume FAST, go back to RUNNING
// ─────────────────────────────────────────────────────────────────
void CarDriver::cmdVelCallback(
    const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
    int64_t stamp = rclcpp::Time(msg->header.stamp).nanoseconds();
    last_ppo_stamp_ns_ = stamp;

    if (!system_ready_ || sync_state_ != SyncState::WAIT_PPO)
        return;

    // Only accept if this is a new command (not a stale one)
    if (stamp <= trigger_stamp_ns_)
        return;

    // Apply command regardless of state (handles non-training mode too)
    double v     = msg->twist.linear.x;
    double omega = msg->twist.angular.z;
    double phi   = 0.0;
    if (std::abs(v) > 0.01) {
        phi = -std::atan2(WHEELBASE * omega, v);
        if (v < 0.0) phi = -phi;
    }
    target_steer_ = std::clamp(phi, -MAX_STEERING, MAX_STEERING);
    target_speed_ = v;
    wbu_driver_set_cruising_speed(v * 3.6);
    wbu_driver_set_steering_angle(target_steer_);

    // Transition: PPO done → resume FAST, go back to RUNNING
    sync_state_ = SyncState::RUNNING;

    if (training_mode_) {
        wb_supervisor_simulation_set_mode(WB_SUPERVISOR_SIMULATION_MODE_FAST);
        RCLCPP_DEBUG(node_->get_logger(),
            "PPO cmd received → FAST (RUNNING)");
    }
}

// ─────────────────────────────────────────────────────────────────
// step() — called every Webots timestep
// ─────────────────────────────────────────────────────────────────
void CarDriver::step()
{
    double current_time = wb_robot_get_time();
    double dt = current_time - last_time_;
    last_time_ = current_time;

    // Always publish sensors
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
    if (current_time - last_cam_pub_ >= CAM_PERIOD) {
        publishCamera();
        last_cam_pub_ = current_time;
    }
    publishGroundTruth();

    if (!system_ready_) return;

    // ── State machine ──────────────────────────────────────────────
    switch (sync_state_) {

    case SyncState::RUNNING: {
        // Initialize trigger clock on first entry
        if (last_rl_trigger_ < 0.0)
            last_rl_trigger_ = current_time;

        // Check if 50 ms of sim-time have elapsed
        if (current_time - last_rl_trigger_ < RL_PERIOD)
            break;

        last_rl_trigger_ = current_time;

        // Switch to real-time so MPC can respond at human speed
        if (training_mode_)
            wb_supervisor_simulation_set_mode(WB_SUPERVISOR_SIMULATION_MODE_REAL_TIME);

        // Publish trigger for MPC (and PPO agent to observe)
        std_msgs::msg::Header trig;
        trig.stamp    = node_->get_clock()->now();
        trig.frame_id = std::to_string(current_time);
        rl_trigger_pub_->publish(trig);
        trigger_stamp_ns_ = rclcpp::Time(trig.stamp).nanoseconds();

        sync_state_ = SyncState::WAIT_MPC;

        // ── Phase 1: wait for MPC (real-time) ─────────────────────
        if (training_mode_) {
            bool mpc_ok = spinUntil(
                [this]{ return sync_state_ == SyncState::WAIT_PPO; },
                MPC_TIMEOUT_MS);

            if (!mpc_ok) {
                RCLCPP_WARN(node_->get_logger(),
                    "MPC timeout (%.0d ms) — resuming FAST without PPO step",
                    MPC_TIMEOUT_MS);
                sync_state_ = SyncState::RUNNING;
                wb_supervisor_simulation_set_mode(WB_SUPERVISOR_SIMULATION_MODE_FAST);
                break;
            }
            // mpcCmdCallback already paused the sim and set WAIT_PPO

            // ── Phase 2: wait for PPO (paused) ────────────────────
            bool ppo_ok = spinUntil(
                [this]{ return sync_state_ == SyncState::RUNNING; },
                PPO_TIMEOUT_MS);

            if (!ppo_ok) {
                RCLCPP_WARN(node_->get_logger(),
                    "PPO timeout (%d ms) — resuming with last MPC command",
                    PPO_TIMEOUT_MS);
                sync_state_ = SyncState::RUNNING;
                wb_supervisor_simulation_set_mode(WB_SUPERVISOR_SIMULATION_MODE_FAST);
            }
            // If ppo_ok: cmdVelCallback already set RUNNING and resumed FAST
        }
        break;
    }

    case SyncState::WAIT_MPC:
    case SyncState::WAIT_PPO:
        // Waiting is handled inside the RUNNING case via spinUntil.
        // These states should not be reached from a fresh step() call
        // outside of training mode, but handle gracefully anyway.
        break;
    }
}

// ─────────────────────────────────────────────────────────────────
// Sensor publishers — identical to original, included for completeness
// ─────────────────────────────────────────────────────────────────

void CarDriver::publishImu()
{
    auto now = node_->get_clock()->now();
    const double *quat = wb_inertial_unit_get_quaternion(imu_);
    const double *gyr  = wb_gyro_get_values(gyro_);
    const double *acc  = wb_accelerometer_get_values(accel_);

    sensor_msgs::msg::Imu msg;
    msg.header.stamp    = now;
    msg.header.frame_id = "base_link";

    msg.orientation.x = quat[0];
    msg.orientation.y = quat[1];
    msg.orientation.z = quat[2];
    msg.orientation.w = quat[3];
    msg.orientation_covariance = {1e-6,0,0, 0,1e-6,0, 0,0,1e-6};

    double s_gyr = GYR_STDDEV * GYR_STDDEV;
    msg.angular_velocity.x = gyr[0] + gyro_bias_[0] + gyr_noise_(rng_);
    msg.angular_velocity.y = gyr[1] + gyro_bias_[1] + gyr_noise_(rng_);
    msg.angular_velocity.z = gyr[2] + gyro_bias_[2] + gyr_noise_(rng_);
    msg.angular_velocity_covariance = {s_gyr,0,0, 0,s_gyr,0, 0,0,s_gyr};

    double s_acc = ACC_STDDEV * ACC_STDDEV;
    msg.linear_acceleration.x = acc[0] + accel_bias_[0] + acc_noise_(rng_);
    msg.linear_acceleration.y = acc[1] + accel_bias_[1] + acc_noise_(rng_);
    msg.linear_acceleration.z = acc[2] + accel_bias_[2] + acc_noise_(rng_);
    msg.linear_acceleration_covariance = {s_acc,0,0, 0,s_acc,0, 0,0,s_acc};

    imu_pub_->publish(msg);
}

void CarDriver::publishMag()
{
    auto now = node_->get_clock()->now();
    const double *vals = wb_compass_get_values(mag_);
    double s = MAG_STDDEV * MAG_STDDEV;

    sensor_msgs::msg::MagneticField msg;
    msg.header.stamp    = now;
    msg.header.frame_id = "base_link";
    msg.magnetic_field.x = vals[0] + mag_noise_(rng_);
    msg.magnetic_field.y = vals[1] + mag_noise_(rng_);
    msg.magnetic_field.z = vals[2] + mag_noise_(rng_);
    msg.magnetic_field_covariance = {s,0,0, 0,s,0, 0,0,s};
    mag_pub_->publish(msg);
}

void CarDriver::publishGps()
{
    auto now = node_->get_clock()->now();
    const double *vals = wb_gps_get_values(gps_);
    double s = GPS_STDDEV * GPS_STDDEV;

    sensor_msgs::msg::NavSatFix msg;
    msg.header.stamp    = now;
    msg.header.frame_id = "gps_link";
    msg.status.status   = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
    msg.status.service  = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;
    msg.latitude  = vals[0];
    msg.longitude = vals[1];
    msg.altitude  = vals[2];
    msg.position_covariance = {s,0,0, 0,s,0, 0,0,s};
    msg.position_covariance_type =
        sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN;
    gps_pub_->publish(msg);
}

void CarDriver::publishCamera()
{
    const unsigned char *seg = wb_camera_recognition_get_segmentation_image(camera_);
    if (!seg) return;

    sensor_msgs::msg::Image msg;
    msg.header.stamp    = node_->get_clock()->now();
    msg.header.frame_id = "camera_link";
    msg.width           = cam_width_;
    msg.height          = cam_height_;
    msg.encoding        = "8UC2";
    msg.step            = cam_width_ * 2;
    msg.data.resize(cam_pixels_ * 2, 0);

    for (int i = 0; i < cam_pixels_; ++i) {
    unsigned char r = seg[i * 4 + 2];  // R: curbs
    unsigned char g = seg[i * 4 + 1];  // G: speedbumps

    msg.data[i * 2 + 0] = (r > 128) ? 255 : 0;
    msg.data[i * 2 + 1] = (g > 128) ? 255 : 0;
}
    seg_pub_->publish(msg);
}

void CarDriver::publishGroundTruth()
{
    auto now = node_->get_clock()->now();
    const double *pos = wb_supervisor_node_get_position(self_node_);
    const double *rot = wb_supervisor_node_get_orientation(self_node_);

    tf2::Matrix3x3 mat(
        rot[0], rot[1], rot[2],
        rot[3], rot[4], rot[5],
        rot[6], rot[7], rot[8]);

    tf2::Quaternion q;
    mat.getRotation(q);

    double roll, pitch, yaw;
    mat.getRPY(roll, pitch, yaw);

    const double *vel = wb_supervisor_node_get_velocity(self_node_);
    double c   =  std::cos(yaw);
    double s_y = std::sin(yaw);
    double vx_b =  c   * vel[0] + s_y * vel[1];
    double vy_b = -s_y * vel[0] + c   * vel[1];

    nav_msgs::msg::Odometry msg;
    msg.header.stamp         = now;
    msg.header.frame_id      = "map";
    msg.child_frame_id       = "base_link";
    msg.pose.pose.position.x = pos[0];
    msg.pose.pose.position.y = pos[1];
    msg.pose.pose.position.z = pos[2];
    msg.pose.pose.orientation.x = q.x();
    msg.pose.pose.orientation.y = q.y();
    msg.pose.pose.orientation.z = q.z();
    msg.pose.pose.orientation.w = q.w();
    msg.twist.twist.linear.x  = vx_b;
    msg.twist.twist.linear.y  = vy_b;
    msg.twist.twist.linear.z  = vel[2];
    msg.twist.twist.angular.x = vel[3];
    msg.twist.twist.angular.y = vel[4];
    msg.twist.twist.angular.z = vel[5];
    gt_pub_->publish(msg);
}

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
    msg.name     = {"left_rear_wheel","right_rear_wheel","left_steer","right_steer"};
    msg.position = {lp, rp, pl, pr};
    msg.velocity = {lv, rv, 0.0, 0.0};
    js_pub_->publish(msg);
    return {lv, rv, pl, pr};
}

void CarDriver::updateOdometry(double lv, double rv,
                               double phi_l, double phi_r, double dt)
{
    if (dt <= 0.0) return;

    auto phi_from = [](double phi, double sign) -> double {
        if (std::abs(phi) < 1e-6) return 0.0;
        return std::atan2(WHEELBASE * std::tan(phi),
                          WHEELBASE + sign * (TRACK_FRONT / 2.0) * std::tan(phi));
    };
    double phi = -0.5 * (phi_from(phi_l, 1.0) + phi_from(phi_r, -1.0));

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

    nav_msgs::msg::Odometry odom;
    odom.header.stamp    = now;
    odom.header.frame_id = "odom";
    odom.child_frame_id  = "base_link";
    odom.pose.pose.position.x    = x_;
    odom.pose.pose.position.y    = y_;
    odom.pose.pose.orientation.z = std::sin(theta_ / 2.0);
    odom.pose.pose.orientation.w = std::cos(theta_ / 2.0);
    odom.twist.twist.linear.x    = v_bx;
    odom.twist.twist.angular.z   = omega_bz;
    odom.pose.covariance[0]  = 0.001;
    odom.pose.covariance[7]  = 0.001;
    odom.pose.covariance[14] = 1.0;
    odom.pose.covariance[21] = 1.0;
    odom.pose.covariance[28] = 1.0;
    odom.pose.covariance[35] = 0.005;
    odom.twist.covariance[0]  = 0.001;
    odom.twist.covariance[7]  = 1.0;
    odom.twist.covariance[14] = 1.0;
    odom.twist.covariance[21] = 1.0;
    odom.twist.covariance[28] = 1.0;
    odom.twist.covariance[35] = 0.02;
    odom_pub_->publish(odom);
}

} // namespace vehicle_webots

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(vehicle_webots::CarDriver,
                       webots_ros2_driver::PluginInterface)