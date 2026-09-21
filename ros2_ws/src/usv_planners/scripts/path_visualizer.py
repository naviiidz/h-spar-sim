#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import tf2_ros
import tf2_geometry_msgs
import math

class PathVisualizer(Node):
    def __init__(self):
        super().__init__('path_visualizer')
        
        # Declare parameters
        self.declare_parameter('robot_frame', 'wamv_simple')
        self.declare_parameter('global_frame', 'odom')
        self.declare_parameter('path_color_r', 0.0)
        self.declare_parameter('path_color_g', 1.0)
        self.declare_parameter('path_color_b', 0.0)
        self.declare_parameter('path_color_a', 1.0)
        self.declare_parameter('path_width', 0.5)
        self.declare_parameter('max_path_points', 1000)
        self.declare_parameter('min_distance_threshold', 0.1)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('max_coordinate_value', 1000.0)  # Filter out unreasonable coordinates
        
        # Get parameters
        self.robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value
        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        self.path_color_r = self.get_parameter('path_color_r').get_parameter_value().double_value
        self.path_color_g = self.get_parameter('path_color_g').get_parameter_value().double_value
        self.path_color_b = self.get_parameter('path_color_b').get_parameter_value().double_value
        self.path_color_a = self.get_parameter('path_color_a').get_parameter_value().double_value
        self.path_width = self.get_parameter('path_width').get_parameter_value().double_value
        self.max_path_points = self.get_parameter('max_path_points').get_parameter_value().integer_value
        self.min_distance = self.get_parameter('min_distance_threshold').get_parameter_value().double_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.max_coordinate = self.get_parameter('max_coordinate_value').get_parameter_value().double_value
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.path_pub = self.create_publisher(Path, 'traversed_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'path_markers', 10)
        
        # Subscriber for goal poses (optional - to mark waypoints)
        self.goal_sub = self.create_subscription(
            PoseStamped, 'goal_pose', self.goal_callback, 10)
        
        # Path storage
        self.path = Path()
        self.path.header.frame_id = self.global_frame
        self.goal_markers = []
        self.goal_counter = 0
        
        # Timer for path updates
        timer_period = 1.0 / self.publish_rate  # Convert Hz to seconds
        self.timer = self.create_timer(timer_period, self.update_path)
        
        self.get_logger().info(f"Path Visualizer initialized")
        self.get_logger().info(f"Robot frame: {self.robot_frame}")
        self.get_logger().info(f"Global frame: {self.global_frame}")
        self.get_logger().info(f"Publishing at {self.publish_rate} Hz")
        self.get_logger().info(f"Max coordinate filter: {self.max_coordinate}")
        
        # Clear any existing unreasonable path points
        self.clear_unreasonable_path_points()
        
    def goal_callback(self, msg):
        """Callback for goal poses - adds waypoint markers"""
        self.goal_counter += 1
        
        # Create waypoint marker
        marker = Marker()
        marker.header = msg.header
        marker.ns = "waypoints"
        marker.id = self.goal_counter
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose = msg.pose
        marker.scale.x = 2.0
        marker.scale.y = 2.0
        marker.scale.z = 2.0
        
        # Yellow waypoint markers
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        # Add text marker with goal number
        text_marker = Marker()
        text_marker.header = msg.header
        text_marker.ns = "waypoint_text"
        text_marker.id = self.goal_counter
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        text_marker.pose = msg.pose
        text_marker.pose.position.z += 3.0  # Lift text above marker
        text_marker.scale.z = 2.0
        text_marker.text = f"Goal {self.goal_counter}"
        
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        
        self.goal_markers.extend([marker, text_marker])
        
        self.get_logger().info(f"Added waypoint marker for goal {self.goal_counter}")
        
    def get_robot_pose(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame, self.robot_frame, rclpy.time.Time())
            
            pose = PoseStamped()
            pose.header.frame_id = self.global_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            
            return pose
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().debug(f"Could not get robot pose: {e}")
            return None
    
    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        dx = pose1.pose.position.x - pose2.pose.position.x
        dy = pose1.pose.position.y - pose2.pose.position.y
        dz = pose1.pose.position.z - pose2.pose.position.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def is_reasonable_position(self, pose):
        """Check if the position coordinates are reasonable (not extremely large)"""
        x = abs(pose.pose.position.x)
        y = abs(pose.pose.position.y)
        z = abs(pose.pose.position.z)
        
        return (x < self.max_coordinate and 
                y < self.max_coordinate and 
                z < self.max_coordinate)
    
    def clear_unreasonable_path_points(self):
        """Remove any existing path points with unreasonable coordinates"""
        if len(self.path.poses) > 0:
            reasonable_poses = []
            removed_count = 0
            
            for pose in self.path.poses:
                if self.is_reasonable_position(pose):
                    reasonable_poses.append(pose)
                else:
                    removed_count += 1
            
            self.path.poses = reasonable_poses
            if removed_count > 0:
                self.get_logger().info(f"Cleared {removed_count} unreasonable path points")
    
    def update_path(self):
        """Timer callback to update the traversed path"""
        current_pose = self.get_robot_pose()
        if current_pose is None:
            return
        
        # Filter out unreasonable positions (like initial spawn coordinates)
        if not self.is_reasonable_position(current_pose):
            self.get_logger().debug(f"Filtering out unreasonable position: x={current_pose.pose.position.x:.1f}, y={current_pose.pose.position.y:.1f}")
            return
        
        # Add pose to path if it's far enough from the last point
        should_add = True
        if len(self.path.poses) > 0:
            distance = self.calculate_distance(current_pose, self.path.poses[-1])
            if distance < self.min_distance:
                should_add = False
        
        if should_add:
            self.path.poses.append(current_pose)
            
            # Limit path length to prevent memory issues
            if len(self.path.poses) > self.max_path_points:
                self.path.poses.pop(0)  # Remove oldest point
        
        # Update path header timestamp
        self.path.header.stamp = self.get_clock().now().to_msg()
        
        # Publish path
        self.path_pub.publish(self.path)
        
        # Create and publish markers
        self.publish_markers()
        
        # Debug info (throttled)
        if len(self.path.poses) % 50 == 0:  # Log every 50 points
            self.get_logger().info(f"Path contains {len(self.path.poses)} points")
    
    def publish_markers(self):
        """Publish visualization markers"""
        marker_array = MarkerArray()
        
        # Path line marker
        if len(self.path.poses) > 1:
            path_marker = Marker()
            path_marker.header = self.path.header
            path_marker.ns = "traversed_path"
            path_marker.id = 0
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            
            path_marker.scale.x = self.path_width
            path_marker.color.r = self.path_color_r
            path_marker.color.g = self.path_color_g
            path_marker.color.b = self.path_color_b
            path_marker.color.a = self.path_color_a
            
            # Add all path points
            for pose in self.path.poses:
                path_marker.points.append(pose.pose.position)
            
            marker_array.markers.append(path_marker)
        
        # Current position marker
        if len(self.path.poses) > 0:
            current_marker = Marker()
            current_marker.header = self.path.header
            current_marker.ns = "current_position"
            current_marker.id = 0
            current_marker.type = Marker.ARROW
            current_marker.action = Marker.ADD
            
            current_marker.pose = self.path.poses[-1].pose
            current_marker.scale.x = 3.0
            current_marker.scale.y = 1.0
            current_marker.scale.z = 1.0
            
            # Blue arrow for current position
            current_marker.color.r = 0.0
            current_marker.color.g = 0.0
            current_marker.color.b = 1.0
            current_marker.color.a = 1.0
            
            marker_array.markers.append(current_marker)
        
        # Start position marker
        if len(self.path.poses) > 0:
            start_marker = Marker()
            start_marker.header = self.path.header
            start_marker.ns = "start_position"
            start_marker.id = 0
            start_marker.type = Marker.CYLINDER
            start_marker.action = Marker.ADD
            
            start_marker.pose = self.path.poses[0].pose
            start_marker.scale.x = 2.0
            start_marker.scale.y = 2.0
            start_marker.scale.z = 0.5
            
            # Green cylinder for start position
            start_marker.color.r = 0.0
            start_marker.color.g = 1.0
            start_marker.color.b = 0.0
            start_marker.color.a = 0.8
            
            marker_array.markers.append(start_marker)
        
        # Add goal markers
        marker_array.markers.extend(self.goal_markers)
        
        # Publish markers
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    
    path_visualizer = PathVisualizer()
    
    try:
        rclpy.spin(path_visualizer)
    except KeyboardInterrupt:
        pass
    
    path_visualizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
