#!/usr/bin/env python3
"""
Drag Force Query Node

Subscribes to vehicle pose and queries the drag_query service to get hydrodynamic
drag forces, then publishes them as wrench commands to Gazebo.

Subscriptions:
  - /wamv/simple_pose (geometry_msgs/PoseStamped): Vehicle pose

Service Calls:
  - /get_drag_force (drag_query/srv/GetDragForce): Query drag forces at current position

Publications:
  - /h_spar/wamv/wrench_cmd (geometry_msgs/WrenchStamped): Wrench command for Gazebo
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, WrenchStamped
from drag_query.srv import GetDragForce
from rosgraph_msgs.msg import Clock
import math


class DragForceNode(Node):
    def __init__(self):
        super().__init__('drag_force_node')

        # Parameters
        self.declare_parameter('pose_topic', '/wamv/simple_pose')
        self.declare_parameter('wrench_topic', '/h_spar/wamv/wrench_cmd')
        self.declare_parameter('drag_query_service', '/get_drag_force')
        self.declare_parameter('clock_topic', '/clock')
        self.declare_parameter('timestamp_period', 521.0)
        self.declare_parameter('publish_rate', 1.0)

        pose_topic = self.get_parameter('pose_topic').value
        wrench_topic = self.get_parameter('wrench_topic').value
        drag_query_service = self.get_parameter('drag_query_service').value
        clock_topic = self.get_parameter('clock_topic').value
        publish_rate = self.get_parameter('publish_rate').value

        # Subscriber to pose
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.pose_callback,
            10
        )

        # Subscription to simulation time
        self.clock_subscription = self.create_subscription(
            Clock,
            clock_topic,
            self.clock_callback,
            10
        )

        # Publisher for wrench
        self.wrench_publisher = self.create_publisher(
            WrenchStamped,
            wrench_topic,
            10
        )

        # Service client for drag query
        self.drag_query_client = self.create_client(
            GetDragForce,
            drag_query_service
        )

        while not self.drag_query_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for drag_query service...')

        self.get_logger().info(
            f'Drag force node initialized. '
            f'Pose: {pose_topic}, Wrench: {wrench_topic}'
        )

        # State
        self.current_pose = None
        self.current_sim_time = None
        self.timestamp_period = int(self.get_parameter('timestamp_period').value)
        self.last_request_time = self.get_clock().now()

        # Timer for publishing
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def pose_callback(self, msg: PoseStamped):
        """Update current pose from subscription."""
        self.current_pose = msg

    def clock_callback(self, msg: Clock):
        """Update current simulation time from /clock."""
        self.current_sim_time = float(msg.clock.sec) + float(msg.clock.nanosec) / 1e9

    def timer_callback(self):
        """Periodically query drag forces and publish wrench."""
        if self.current_pose is None:
            self.get_logger().debug('Waiting for pose...')
            return

        if self.current_sim_time is None:
            self.get_logger().debug('Waiting for /clock...')
            return

        # Extract pose data
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y

        # Extract heading (yaw) from quaternion
        quat = self.current_pose.pose.orientation
        theta = self.quaternion_to_yaw(quat.x, quat.y, quat.z, quat.w)

        # Map simulation time into the available velocity-field window.
        timestamp = int(self.current_sim_time)
        if self.timestamp_period > 0:
            timestamp = timestamp % self.timestamp_period

        self.get_logger().debug(f'Timer called: timestamp={timestamp}, x={x:.2f}, y={y:.2f}, theta={theta:.2f}')
        
        # Query drag forces
        self.query_drag_force(timestamp, x, y, theta)

    def quaternion_to_yaw(self, x, y, z, w):
        """Convert quaternion to yaw angle."""
        sin_roll = 2 * (w * x + y * z)
        cos_roll = 1 - 2 * (x * x + y * y)
        sin_pitch = 2 * (w * y - z * x)
        sin_yaw = 2 * (w * z + x * y)
        cos_yaw = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(sin_yaw, cos_yaw)
        return yaw

    def query_drag_force(self, timestamp, x, y, theta):
        """
        Query the drag_query service for drag forces at the current position.

        Args:
            timestamp: Time (seconds)
            x: X position (meters)
            y: Y position (meters)
            theta: Heading angle (radians)
        """
        self.get_logger().debug(f'Querying drag force at x={x:.2f}, y={y:.2f}')
        
        request = GetDragForce.Request()
        request.timestamp = float(timestamp)
        request.x = x
        request.y = y
        request.theta = theta

        # Async service call to avoid blocking the event loop
        future = self.drag_query_client.call_async(request)
        future.add_done_callback(self.drag_force_response_callback)

    def drag_force_response_callback(self, future):
        """Handle response from drag_query service."""
        try:
            response = future.result()
            self.get_logger().info(f'Got drag response: fx={response.fx:.2f}, fy={response.fy:.2f}')

            # Create wrench message
            wrench_msg = WrenchStamped()
            wrench_msg.header.stamp = self.get_clock().now().to_msg()
            wrench_msg.header.frame_id = 'wamv/base_link'

            # Set forces from drag query response
            wrench_msg.wrench.force.x = -response.fx
            wrench_msg.wrench.force.y = -response.fy
            wrench_msg.wrench.force.z = 0.0

            # Set torques (currently disabled in drag_query)
            wrench_msg.wrench.torque.x = 0.0
            wrench_msg.wrench.torque.y = 0.0
            wrench_msg.wrench.torque.z = response.torque

            # Publish wrench
            self.wrench_publisher.publish(wrench_msg)
            self.get_logger().info('Published wrench command')

        except Exception as e:
            self.get_logger().error(f'Drag query service failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = DragForceNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
