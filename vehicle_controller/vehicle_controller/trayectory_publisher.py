#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler


class TrajectoryPublisher(Node):

    def __init__(self):

        super().__init__('trajectory_publisher')

        self.pub = self.create_publisher(Path, '/trajectory', 10)

        self.timer = self.create_timer(1.0, self.publish_path)

        # Trajectory parameters
        self.traj_type = 'figure8'  # 'circle', 'straight', 'figure8'
        
        # Circle trajectory parameters
        self.radius = 6.0
        self.center_x = 6.0
        self.center_y = 0.0
        
        # Straight-line trajectory parameters
        self.length = 10.0
        
        self.points = 1000

    def publish_path(self):

        path = Path()
        path.header.frame_id = "odom"  # Fixed reference frame
        path.header.stamp = self.get_clock().now().to_msg()

        if self.traj_type == 'circle':
            poses = self.generate_circle()
        elif self.traj_type == 'straight':
            poses = self.generate_straight_line()
        elif self.traj_type == 'figure8':
            poses = self.generate_figure8()
        else:
            poses = self.generate_straight_line()

        path.poses = poses
        self.pub.publish(path)
        
        self.get_logger().info(
            f'Published {self.traj_type} trajectory with {len(poses)} points',
            throttle_duration_sec=5.0
        )

        raise SystemExit

    # Straight line 
    def generate_straight_line(self):
        """Straight line in the +X direction."""
        poses = []
        
        x_vals = np.linspace(0, self.length, self.points)
        
        for x in x_vals:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            
            pose.pose.position.x = x
            pose.pose.position.y = 0.0
            pose.pose.position.z = 0.0
            
            quat = quaternion_from_euler(0, 0, 0)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            
            poses.append(pose)
        
        return poses

    # Circle 
    def generate_circle(self):
        poses = []
        
        # Center at (0, 6); the path passes through the origin.
        center_x = 0.0
        center_y = self.radius  # 6.0
        
        theta = np.linspace(0, 2*np.pi, self.points, endpoint=False)
        
        for t in theta:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            
            x = center_x + self.radius * np.sin(t)
            y = center_y - self.radius * np.cos(t)
            
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            
            # At t=0, yaw=0 and the heading points toward +X.
            yaw = t
            
            quat = quaternion_from_euler(0, 0, yaw)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            
            poses.append(pose)
        
        return poses


    # Figure-eight trajectory (lemniscate)
    def generate_figure8(self):
        """Figure-eight trajectory."""
        poses = []
        
        t_vals = np.linspace(0, 2*np.pi, self.points)
        scale = 4.0
        
        for t in t_vals:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            
            # Parametric lemniscate
            x = scale * np.sin(t)
            y = scale * np.sin(t) * np.cos(t)
            
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            
            # Compute orientation from the path derivative
            dx = scale * np.cos(t)
            dy = scale * (np.cos(2*t))
            yaw = np.arctan2(dy, dx)
            
            quat = quaternion_from_euler(0, 0, yaw)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            
            poses.append(pose)
        
        return poses


def main():
    rclpy.init()
    node = TrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()