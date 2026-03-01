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

        # Parámetros de trayectoria
        self.traj_type = 'figure8'  # 'circle', 'straight', 'figure8'
        
        # Para círculo
        self.radius = 6.0
        self.center_x = 6.0  # Robot empieza en (0,0), círculo en (3,0)
        self.center_y = 0.0
        
        # Para línea recta
        self.length = 10.0
        
        self.points = 1000

    def publish_path(self):

        path = Path()
        path.header.frame_id = "odom"  # ✅ Frame fijo
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

    # ==========================================
    # Línea recta (robot empieza en origen)
    # ==========================================
    def generate_straight_line(self):
        """Línea recta en dirección +X"""
        poses = []
        
        x_vals = np.linspace(0, self.length, self.points)
        
        for x in x_vals:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            
            pose.pose.position.x = x
            pose.pose.position.y = 0.0
            pose.pose.position.z = 0.0
            
            # Orientación apuntando hacia +X (yaw = 0)
            quat = quaternion_from_euler(0, 0, 0)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            
            poses.append(pose)
        
        return poses

    # ==========================================
    # Círculo (robot empieza cerca del inicio)
    # ==========================================
    def generate_circle(self):
        poses = []
        
        # ✅ Centro en (0, 6) → pasa exactamente por (0,0)
        center_x = 0.0
        center_y = self.radius  # 6.0
        
        theta = np.linspace(0, 2*np.pi, self.points, endpoint=False)
        
        for t in theta:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            
            # t=0 → x=0, y=0 ✅
            x = center_x + self.radius * np.sin(t)
            y = center_y - self.radius * np.cos(t)
            
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            
            # t=0 → yaw=0 (apunta hacia +X) ✅
            yaw = t
            
            quat = quaternion_from_euler(0, 0, yaw)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            
            poses.append(pose)
        
        return poses

    # ==========================================
    # Figura de 8 (lemniscata)
    # ==========================================
    def generate_figure8(self):
        """Trayectoria en forma de 8"""
        poses = []
        
        t_vals = np.linspace(0, 2*np.pi, self.points)
        scale = 3.0
        
        for t in t_vals:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            
            # Lemniscata paramétrica
            x = scale * np.sin(t)
            y = scale * np.sin(t) * np.cos(t)
            
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            
            # Calcular orientación a partir de la derivada
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