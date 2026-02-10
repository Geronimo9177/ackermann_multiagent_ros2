#include <cmath>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "control_msgs/msg/multi_dof_command.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"

class AckermannLowLevelController : public rclcpp::Node
{
public:
  AckermannLowLevelController()
  : Node("ackermann_low_level_controller")
  {
    wheelbase_ = this->declare_parameter("wheelbase", 0.335);
    steering_track_width_ = this->declare_parameter("steering_track_width", 0.305);
    traction_track_width_ = this->declare_parameter("traction_track_width", 0.305);
    wheel_radius_ = this->declare_parameter("wheel_radius", 0.073025);
    use_joint_state_odom_ = this->declare_parameter("use_joint_state_odom", true);
    left_steer_joint_ = this->declare_parameter("left_steer_joint", std::string("left_steering_joint"));
    right_steer_joint_ = this->declare_parameter("right_steer_joint", std::string("right_steering_joint"));
    left_rear_joint_ = this->declare_parameter("left_rear_joint", std::string("left_rear_axle"));
    right_rear_joint_ = this->declare_parameter("right_rear_joint", std::string("right_rear_axle"));
    publish_tf_ = this->declare_parameter("publish_tf", true);
    odom_frame_id_ = this->declare_parameter("odom_frame_id", std::string("odom"));
    base_frame_id_ = this->declare_parameter("base_frame_id", std::string("base_link"));

    cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel",
      rclcpp::QoS(10),
      std::bind(&AckermannLowLevelController::cmdCallback, this, std::placeholders::_1));

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "joint_states",
      rclcpp::QoS(50),
      std::bind(&AckermannLowLevelController::jointStateCallback, this, std::placeholders::_1));

    steering_pub_ = this->create_publisher<control_msgs::msg::MultiDOFCommand>(
      "steering_pid/reference",
      rclcpp::QoS(10));

    rear_wheel_pub_ = this->create_publisher<control_msgs::msg::MultiDOFCommand>(
      "rear_wheel_pid/reference",
      rclcpp::QoS(10));

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", rclcpp::QoS(10));

    if (publish_tf_) {
      tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    }

    last_odom_time_ = this->now();

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&AckermannLowLevelController::update, this));
  }

private:
  void cmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    cmd_v_ = msg->linear.x;
    cmd_w_ = msg->angular.z;
  }

  void update()
  {
    const auto now = this->now();
    double v = cmd_v_;
    double w = cmd_w_;


    double left_steer = 0.0;
    double right_steer = 0.0;
    double left_wheel_vel = 0.0;
    double right_wheel_vel = 0.0;

    const double eps = 1e-6;
    if (std::fabs(w) < eps || std::fabs(v) < eps) {
      left_steer = 0.0;
      right_steer = 0.0;
      left_wheel_vel = v / wheel_radius_;
      right_wheel_vel = v / wheel_radius_;
    } else {
      const double R = v / w;
      const double half_steer_track = steering_track_width_ * 0.5;
      const double half_trac_track = traction_track_width_ * 0.5;

      const double R_left = R - half_steer_track;
      const double R_right = R + half_steer_track;

      if (std::fabs(R_left) > eps && std::fabs(R_right) > eps) {
        left_steer = std::atan(wheelbase_ / R_left);
        right_steer = std::atan(wheelbase_ / R_right);
      }

      const double v_left = w * (R - half_trac_track);
      const double v_right = w * (R + half_trac_track);
      left_wheel_vel = v_left / wheel_radius_;
      right_wheel_vel = v_right / wheel_radius_;
    }

    publishControllers(left_steer, right_steer, left_wheel_vel, right_wheel_vel);

    double v_odom = v;
    double w_odom = w;
    if (use_joint_state_odom_ && have_joint_states_) {
      const double v_left = left_rear_vel_ * wheel_radius_;
      const double v_right = right_rear_vel_ * wheel_radius_;
      v_odom = 0.5 * (v_left + v_right);

      const double steer_avg = 0.5 * (left_steer_pos_ + right_steer_pos_);
      if (std::fabs(steer_avg) > 1e-6) {
        w_odom = v_odom * std::tan(steer_avg) / wheelbase_;
      } else {
        w_odom = 0.0;
      }
    }

    publishOdometry(v_odom, w_odom, now);
  }

  void publishOdometry(double v, double w, const rclcpp::Time & now)
  {
    const double dt = (now - last_odom_time_).seconds();
    if (dt <= 0.0) {
      return;
    }

    x_ += v * std::cos(yaw_) * dt;
    y_ += v * std::sin(yaw_) * dt;
    yaw_ += w * dt;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw_);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = now;
    odom.header.frame_id = odom_frame_id_;
    odom.child_frame_id = base_frame_id_;
    odom.pose.pose.position.x = x_;
    odom.pose.pose.position.y = y_;
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom.twist.twist.linear.x = v;
    odom.twist.twist.angular.z = w;

    odom_pub_->publish(odom);

    if (publish_tf_ && tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header.stamp = now;
      tf_msg.header.frame_id = odom_frame_id_;
      tf_msg.child_frame_id = base_frame_id_;
      tf_msg.transform.translation.x = x_;
      tf_msg.transform.translation.y = y_;
      tf_msg.transform.translation.z = 0.0;
      tf_msg.transform.rotation.x = q.x();
      tf_msg.transform.rotation.y = q.y();
      tf_msg.transform.rotation.z = q.z();
      tf_msg.transform.rotation.w = q.w();
      tf_broadcaster_->sendTransform(tf_msg);
    }

    last_odom_time_ = now;
  }

  void publishControllers(
    double left_steer,
    double right_steer,
    double left_wheel_vel,
    double right_wheel_vel)
  {
    control_msgs::msg::MultiDOFCommand steering_msg;
    steering_msg.dof_names = {left_steer_joint_, right_steer_joint_};
    steering_msg.values = {left_steer, right_steer};
    steering_pub_->publish(steering_msg);

    control_msgs::msg::MultiDOFCommand ref_msg;
    ref_msg.dof_names = {left_rear_joint_, right_rear_joint_};
    ref_msg.values = {left_wheel_vel, right_wheel_vel};
    rear_wheel_pub_->publish(ref_msg);
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    for (size_t i = 0; i < msg->name.size(); ++i) {
      const auto & name = msg->name[i];
      if (name == left_steer_joint_) {
        if (i < msg->position.size()) {
          left_steer_pos_ = msg->position[i];
        }
      } else if (name == right_steer_joint_) {
        if (i < msg->position.size()) {
          right_steer_pos_ = msg->position[i];
        }
      } else if (name == left_rear_joint_) {
        if (i < msg->velocity.size()) {
          left_rear_vel_ = msg->velocity[i];
        }
        if (i < msg->position.size()) {
          left_rear_pos_ = msg->position[i];
        }
      } else if (name == right_rear_joint_) {
        if (i < msg->velocity.size()) {
          right_rear_vel_ = msg->velocity[i];
        }
        if (i < msg->position.size()) {
          right_rear_pos_ = msg->position[i];
        }
      }
    }

    have_joint_states_ = true;
  }

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Publisher<control_msgs::msg::MultiDOFCommand>::SharedPtr steering_pub_;
  rclcpp::Publisher<control_msgs::msg::MultiDOFCommand>::SharedPtr rear_wheel_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;

  double wheelbase_{0.335};
  double steering_track_width_{0.305};
  double traction_track_width_{0.305};
  double wheel_radius_{0.073025};
  bool use_joint_state_odom_{true};
  std::string left_steer_joint_{"left_steering_joint"};
  std::string right_steer_joint_{"right_steering_joint"};
  std::string left_rear_joint_{"left_rear_axle"};
  std::string right_rear_joint_{"right_rear_axle"};
  bool publish_tf_{true};
  std::string odom_frame_id_{"odom"};
  std::string base_frame_id_{"base_link"};

  rclcpp::Time last_odom_time_;

  double cmd_v_{0.0};
  double cmd_w_{0.0};

  bool have_joint_states_{false};
  double left_steer_pos_{0.0};
  double right_steer_pos_{0.0};
  double left_rear_vel_{0.0};
  double right_rear_vel_{0.0};
  double left_rear_pos_{0.0};
  double right_rear_pos_{0.0};

  double x_{0.0};
  double y_{0.0};
  double yaw_{0.0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AckermannLowLevelController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
