#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile
import time

class FollowPathClient(Node):
    def __init__(self):
        super().__init__('follow_path_client')
        self.action_client = ActionClient(self, FollowPath, '/shelfino0/follow_path')
        self.path_subscription = self.create_subscription(
            Path,
            '/shelfino0/plan1',
            self.path_callback,
            QoSProfile(depth=10)
        )
        self.path_received = False
        self.path_msg = None
        self.get_logger().info('Follow Path Action Client Node has been started.')

    def path_callback(self, msg):
        if not self.path_received:
            self.path_msg = msg
            self.path_received = True
            self.get_logger().info('Received Path message. Sending FollowPath action...')
            self.send_follow_path_goal()

    def send_follow_path_goal(self):
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('FollowPath action server not available!')
            return

        goal_msg = FollowPath.Goal()
        goal_msg.path = self.path_msg

        self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        ).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('FollowPath goal rejected.')
            return

        self.get_logger().info('FollowPath goal accepted.')
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback}')

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        if status == 4:
            self.get_logger().info('FollowPath goal was canceled.')
        elif status == 5:
            self.get_logger().info('FollowPath goal failed.')
        else:
            self.get_logger().info(f'FollowPath result: {result}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    follow_path_client = FollowPathClient()
    try:
        rclpy.spin(follow_path_client)
    except KeyboardInterrupt:
        pass
    finally:
        follow_path_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
