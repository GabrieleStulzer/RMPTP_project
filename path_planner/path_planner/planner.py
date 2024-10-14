#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import math

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.publisher_ = self.create_publisher(Path, '/shelfino0/plan1', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.publish_path)
        self.get_logger().info('Path Publisher Node has been started.')

    def publish_path(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'  # Change as per your TF frame

        # Create sample waypoints (e.g., a circular path)
        num_points = 20
        radius = 5.0
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = radius * math.cos(theta)
            pose.pose.position.y = radius * math.sin(theta)
            pose.pose.position.z = 0.0
            # Orientation can be set as needed; here we keep it zero
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.publisher_.publish(path_msg)
        self.get_logger().info(f'Published Path with {num_points} points.')

def main(args=None):
    rclpy.init(args=args)
    path_publisher = PathPublisher()
    try:
        rclpy.spin(path_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        path_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()