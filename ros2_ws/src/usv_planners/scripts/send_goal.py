#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import sys

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')
        self.publisher = self.create_publisher(PoseStamped, 'goal_pose', 10)
        
    def send_goal(self, x, y, yaw=0.0):
        goal = PoseStamped()
        goal.header.frame_id = 'odom'
        goal.header.stamp = self.get_clock().now().to_msg()
        
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        import math
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.publisher.publish(goal)
        self.get_logger().info(f'Published goal: x={x}, y={y}, yaw={yaw}')

def main(args=None):
    rclpy.init(args=args)
    
    if len(sys.argv) < 3:
        print("Usage: python3 send_goal.py <x> <y> [yaw]")
        print("Example: python3 send_goal.py 10.0 5.0")
        return
    
    x = sys.argv[1]
    y = sys.argv[2]
    yaw = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    
    node = GoalPublisher()
    node.send_goal(x, y, yaw)
    
    # Keep node alive for a moment to ensure message is sent
    rclpy.spin_once(node, timeout_sec=1.0)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
