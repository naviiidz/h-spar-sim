#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import math
import time
import sys
import csv
import os
import glob
import subprocess
from datetime import datetime

class SequentialGoalSender(Node):
    def __init__(self):
        super().__init__('sequential_goal_sender')
        
        # Publishers and Subscribers
        self.goal_publisher = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.status_subscriber = self.create_subscription(
            String, 'navigation_status', self.status_callback, 10)
        
        # Try multiple possible odometry topics
        self.odom_topics = [
            '/wamv/simple_pose'
        ]
        
        self.odom_subscriber = None
        self.setup_odometry_subscription()
        
        # State tracking
        self.current_goal_index = 0
        self.goals = []
        self.goal_sent = False
        self.navigation_complete = False
        self.last_status = ""
        
        # Robot position tracking
        self.current_position = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        self.trajectory_log = []  # Log of robot trajectory with timestamps
        self.last_trajectory_record_time = 0
        self.trajectory_record_interval = 0.5  # seconds - record every 0.5 seconds
        self.position_received = False
        self.mission_start_time = None
        self.trajectory_recording_active = False
        
        # Parameters
        self.goal_timeout = 600.0  # seconds to wait for goal completion
        self.goal_start_time = None
        
        # Path saving
        self.execution_log = []  # Log of executed goals with timestamps
        self.start_time = None
        
        # Timer to check goal status
        self.timer = self.create_timer(1.0, self.check_goal_status)
        
        self.get_logger().info("Sequential Goal Sender initialized with position tracking")
    
    def setup_odometry_subscription(self):
        """Try to subscribe to available odometry topics"""
        for topic in self.odom_topics:
            try:
                self.get_logger().info(f"Trying to subscribe to odometry topic: {topic}")
                self.odom_subscriber = self.create_subscription(
                    PoseStamped, topic, self.pose_callback, 10)
                self.get_logger().info(f"Successfully subscribed to odometry topic: {topic}")
                break
            except Exception as e:
                self.get_logger().warn(f"Failed to subscribe to {topic}: {e}")
                continue
        
        if self.odom_subscriber is None:
            self.get_logger().warn("Could not subscribe to any odometry topic. Position tracking disabled.")
    
    def status_callback(self, msg):
        """Callback for navigation status updates"""
        self.last_status = msg.data
        self.get_logger().debug(f"Status update: {msg.data}")
        
        if "GOAL_REACHED" in msg.data or "Target reached" in msg.data:
            self.navigation_complete = True
            self.get_logger().info(f"Goal {self.current_goal_index + 1} completed!")
    
    def pose_callback(self, msg):
        """Callback for robot pose updates (PoseStamped)"""
        # Update current position
        self.current_position['x'] = msg.pose.position.x
        self.current_position['y'] = msg.pose.position.y
        self.current_position['z'] = msg.pose.position.z
        
        # Convert quaternion to yaw
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        
        # Calculate yaw from quaternion
        self.current_position['yaw'] = math.atan2(2.0 * (qw * qz + qx * qy), 
                                                  1.0 - 2.0 * (qy * qy + qz * qz))
        
        if not self.position_received:
            self.position_received = True
            self.get_logger().info(f"First position received: x={self.current_position['x']:.2f}, "
                                  f"y={self.current_position['y']:.2f}")
        
        # Record trajectory if recording is active and enough time has passed
        current_time = time.time()
        if (self.trajectory_recording_active and 
            current_time - self.last_trajectory_record_time >= self.trajectory_record_interval):
            self.record_trajectory_point()
            self.last_trajectory_record_time = current_time
            self.get_logger().debug(f"Auto-recorded trajectory point at interval")

    def odom_callback(self, msg):
        """Callback for robot odometry updates (Odometry message type)"""
        # Update current position
        self.current_position['x'] = msg.pose.pose.position.x
        self.current_position['y'] = msg.pose.pose.position.y
        self.current_position['z'] = msg.pose.pose.position.z
        
        # Convert quaternion to yaw
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        # Calculate yaw from quaternion
        self.current_position['yaw'] = math.atan2(2.0 * (qw * qz + qx * qy), 
                                                  1.0 - 2.0 * (qy * qy + qz * qz))
        
        if not self.position_received:
            self.position_received = True
            self.get_logger().info(f"First position received: x={self.current_position['x']:.2f}, "
                                  f"y={self.current_position['y']:.2f}")
        
        # Record trajectory if recording is active and enough time has passed
        current_time = time.time()
        if (self.trajectory_recording_active and 
            current_time - self.last_trajectory_record_time >= self.trajectory_record_interval):
            self.record_trajectory_point()
            self.last_trajectory_record_time = current_time
            self.get_logger().debug(f"Auto-recorded trajectory point at interval")
    
    def record_trajectory_point(self):
        """Record current robot position as part of trajectory"""
        current_time = time.time()
        
        # Calculate mission elapsed time
        mission_elapsed = current_time - self.mission_start_time if self.mission_start_time else 0
        
        # Calculate distance from current goal if available
        distance_to_goal = 0.0
        if self.current_goal_index < len(self.goals):
            goal = self.goals[self.current_goal_index]
            distance_to_goal = math.sqrt(
                (self.current_position['x'] - goal['x'])**2 + 
                (self.current_position['y'] - goal['y'])**2
            )
        
        trajectory_entry = {
            'timestamp': datetime.fromtimestamp(current_time).isoformat(),
            'mission_time_sec': mission_elapsed,
            'x_ned': self.current_position['x'],
            'y_ned': self.current_position['y'],
            'z_ned': self.current_position['z'],
            'yaw_rad': self.current_position['yaw'],
            'yaw_deg': math.degrees(self.current_position['yaw']),
            'current_goal_number': self.current_goal_index + 1 if self.current_goal_index < len(self.goals) else 'completed',
            'distance_to_current_goal': distance_to_goal,
            'navigation_status': self.last_status
        }
        self.trajectory_log.append(trajectory_entry)
        
        self.get_logger().info(f"Trajectory point {len(self.trajectory_log)} recorded: x={self.current_position['x']:.2f}, "
                               f"y={self.current_position['y']:.2f}, goal={trajectory_entry['current_goal_number']}")
    
    def add_goal(self, x, y, yaw=0.0):
        """Add a goal to the sequence"""
        goal = {
            'x': float(x),
            'y': float(y), 
            'yaw': float(yaw)
        }
        self.goals.append(goal)
        self.get_logger().info(f"Added goal: x={x}, y={y}, yaw={yaw}")
    
    def save_goals_to_csv(self, filename=None):
        """Save the goal sequence to CSV file"""
        if not self.goals:
            self.get_logger().warn("No goals to save")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"goal_sequence_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['goal_number', 'x', 'y', 'yaw_rad', 'yaw_deg'])
                
                for i, goal in enumerate(self.goals):
                    writer.writerow([
                        i + 1,
                        goal['x'],
                        goal['y'], 
                        goal['yaw'],
                        math.degrees(goal['yaw'])
                    ])
            
            self.get_logger().info(f"Goals saved to {filename} with {len(self.goals)} points")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to save goals: {e}")
            return False
    
    def save_execution_log_to_csv(self, filename=None):
        """Save the execution log with timestamps to CSV file"""
        if not self.execution_log:
            self.get_logger().warn("No execution log to save")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"execution_log_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['goal_number', 'x', 'y', 'yaw_rad', 'yaw_deg', 'start_time', 'end_time', 'duration_sec', 'status'])
                
                for log_entry in self.execution_log:
                    writer.writerow([
                        log_entry['goal_number'],
                        log_entry['x'],
                        log_entry['y'],
                        log_entry['yaw'],
                        math.degrees(log_entry['yaw']),
                        log_entry['start_time'],
                        log_entry['end_time'],
                        log_entry['duration'],
                        log_entry['status']
                    ])
            
            self.get_logger().info(f"Execution log saved to {filename} with {len(self.execution_log)} entries")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to save execution log: {e}")
            return False
    
    def save_path_to_csv(self, filename=None):
        """Save the recorded robot path to CSV file"""
        if not self.trajectory_log:
            self.get_logger().warn("No path data to save")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"robot_path_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'time_sec', 'x_ned', 'y_ned', 'z_ned', 'yaw_rad', 'yaw_deg', 'current_goal'])
                
                for entry in self.trajectory_log:
                    writer.writerow([
                        entry['timestamp'],
                        entry['mission_time_sec'],
                        entry['x_ned'],
                        entry['y_ned'],
                        entry['z_ned'],
                        entry['yaw_rad'],
                        math.degrees(entry['yaw_rad']),
                        entry['current_goal_number']
                    ])
            
            self.get_logger().info(f"Robot path saved to {filename} with {len(self.trajectory_log)} position points")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to save robot path: {e}")
            return False
    
    def save_trajectory_to_csv(self, filename=None):
        """Save the recorded robot trajectory to CSV file"""
        if not self.trajectory_log:
            self.get_logger().warn("No trajectory data to save")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"robot_trajectory_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 'mission_time_sec', 'x_ned', 'y_ned', 'z_ned', 'yaw_rad', 'yaw_deg', 
                    'current_goal_number', 'distance_to_current_goal', 'navigation_status'
                ])
                
                for entry in self.trajectory_log:
                    writer.writerow([
                        entry['timestamp'],
                        round(entry['mission_time_sec'], 2),
                        round(entry['x_ned'], 3),
                        round(entry['y_ned'], 3),
                        round(entry['z_ned'], 3),
                        round(entry['yaw_rad'], 4),
                        round(entry['yaw_deg'], 2),
                        entry['current_goal_number'],
                        round(entry['distance_to_current_goal'], 3),
                        entry['navigation_status']
                    ])
            
            self.get_logger().info(f"Robot trajectory saved to {filename} with {len(self.trajectory_log)} points")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to save robot trajectory: {e}")
            return False
    
    def load_astar_waypoints_from_csv(self, csv_file=None):
        """Load waypoints from A* exported CSV file"""
        import glob  # Move import to top of method
        
        if csv_file is None:
            # Look for the most recent A* waypoint file
            pattern = "astar_waypoints_*.csv"
            files = glob.glob(pattern)
            
            if not files:
                self.get_logger().error(f"No A* waypoint files found with pattern {pattern}")
                self.get_logger().info("Please export waypoints first using:")
                self.get_logger().info("  python3 src/usv_sampling/scripts/export_waypoints_with_intervals.py")
                return False
            
            # Get the most recent file
            csv_file = max(files, key=os.path.getctime)
            self.get_logger().info(f"Using most recent A* waypoint file: {csv_file}")
        
        try:
            # First try relative to current directory, then try waypoints subdirectory
            if not os.path.isfile(csv_file):
                waypoints_path = os.path.join("waypoints", csv_file)
                if os.path.isfile(waypoints_path):
                    csv_file = waypoints_path
                else:
                    # Try absolute path
                    csv_file = os.path.join(os.getcwd(), csv_file)
            
            with open(csv_file, 'r') as file:
                # Skip comment lines starting with #
                lines = []
                for line in file:
                    if not line.strip().startswith('#') and line.strip():
                        lines.append(line)
                
                if not lines:
                    self.get_logger().error(f"No valid data lines found in {csv_file}")
                    return False
                
                # Create a string buffer from the non-comment lines
                import io
                csv_data = io.StringIO(''.join(lines))
                reader = csv.DictReader(csv_data)
                waypoint_count = 0
                
                for row in reader:
                    # Skip rows without required columns
                    if 'x' not in row or 'y' not in row:
                        continue
                        
                    try:
                        # Convert quaternion to yaw if available, otherwise use 0
                        yaw = 0.0
                        if 'qz' in row and 'qw' in row and row['qz'] and row['qw']:
                            # Extract yaw from quaternion (assuming 2D navigation)
                            qz = float(row['qz'])
                            qw = float(row['qw'])
                            yaw = 2.0 * math.atan2(qz, qw)
                        
                        # Add waypoint to goals
                        self.add_goal(
                            float(row['x']), 
                            float(row['y']), 
                            yaw
                        )
                        waypoint_count += 1
                    except (ValueError, KeyError) as e:
                        self.get_logger().warn(f"Skipping invalid waypoint row: {row}, error: {e}")
                        continue
                
                self.get_logger().info(f"Loaded {waypoint_count} A* waypoints from {csv_file}")
                return True
                
        except FileNotFoundError:
            self.get_logger().error(f"A* waypoint file not found: {csv_file}")
            self.get_logger().info("Available files:")
            for f in glob.glob("astar_waypoints_*.csv"):
                self.get_logger().info(f"  {f}")
            return False
            
        except Exception as e:
            self.get_logger().error(f"Error loading A* waypoints: {e}")
            return False
    
    def create_pose_stamped(self, x, y, yaw):
        """Create a PoseStamped message from x, y, yaw"""
        goal = PoseStamped()
        goal.header.frame_id = 'odom'
        goal.header.stamp = self.get_clock().now().to_msg()
        
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        
        return goal
    
    def send_current_goal(self):
        """Send the current goal in the sequence"""
        if self.current_goal_index >= len(self.goals):
            self.get_logger().info("All goals completed!")
            return False
        
        goal_data = self.goals[self.current_goal_index]
        goal_msg = self.create_pose_stamped(
            goal_data['x'], 
            goal_data['y'], 
            goal_data['yaw']
        )
        
        self.goal_publisher.publish(goal_msg)
        self.goal_sent = True
        self.navigation_complete = False
        self.goal_start_time = time.time()
        
        # Record trajectory point when goal is sent
        if self.trajectory_recording_active and self.position_received:
            self.record_trajectory_point()
            self.get_logger().info("🔴 Recorded trajectory point when goal sent")
        else:
            self.get_logger().warn(f"🔴 Cannot record trajectory: recording_active={self.trajectory_recording_active}, position_received={self.position_received}")
        
        # Log goal start
        if self.start_time is None:
            self.start_time = time.time()
        
        self.get_logger().info(
            f"Sent goal {self.current_goal_index + 1}/{len(self.goals)}: "
            f"x={goal_data['x']}, y={goal_data['y']}, yaw={goal_data['yaw']}"
        )
        return True
    
    def check_goal_status(self):
        """Timer callback to check goal completion and send next goal"""
        if not self.goals:
            return
        
        # Send first goal if none sent yet
        if not self.goal_sent:
            if not self.send_current_goal():
                return
        
        # Check if current goal is complete
        if self.navigation_complete:
            # Record final trajectory point for this goal
            if self.trajectory_recording_active and self.position_received:
                self.record_trajectory_point()
            
            # Log completed goal
            end_time = time.time()
            goal_data = self.goals[self.current_goal_index]
            duration = end_time - self.goal_start_time if self.goal_start_time else 0
            
            log_entry = {
                'goal_number': self.current_goal_index + 1,
                'x': goal_data['x'],
                'y': goal_data['y'],
                'yaw': goal_data['yaw'],
                'start_time': datetime.fromtimestamp(self.goal_start_time).isoformat() if self.goal_start_time else '',
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'duration': duration,
                'status': 'COMPLETED'
            }
            self.execution_log.append(log_entry)
            
            self.current_goal_index += 1
            self.goal_sent = False
            self.navigation_complete = False
            
            # Small delay before next goal
            time.sleep(2.0)
            
            # Send next goal or finish
            if self.current_goal_index < len(self.goals):
                self.send_current_goal()
                return
            else:
                self.get_logger().info("🎉 All goals completed successfully!")
                subprocess.run(["pkill", "-f", "ros2"], check=False)
                # Stop trajectory recording
                self.trajectory_recording_active = False
                
                # Save all logs before shutdown
                self.save_execution_log_to_csv()
                if self.trajectory_log:
                    self.save_trajectory_to_csv()
                    self.get_logger().info(f"Mission completed! Recorded {len(self.trajectory_log)} trajectory points over {len(self.goals)} goals")
                else:
                    self.get_logger().warn("No trajectory data recorded - check odometry topic")
                
                # Proper shutdown
                self.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                return
        
        # Check for timeout
        elif self.goal_start_time and (time.time() - self.goal_start_time) > self.goal_timeout:
            # Log timed out goal
            end_time = time.time()
            goal_data = self.goals[self.current_goal_index]
            duration = end_time - self.goal_start_time
            
            log_entry = {
                'goal_number': self.current_goal_index + 1,
                'x': goal_data['x'],
                'y': goal_data['y'],
                'yaw': goal_data['yaw'],
                'start_time': datetime.fromtimestamp(self.goal_start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'duration': duration,
                'status': 'TIMEOUT'
            }
            self.execution_log.append(log_entry)
            
            self.get_logger().warn(
                f"Goal {self.current_goal_index + 1} timed out after {self.goal_timeout}s. "
                f"Moving to next goal..."
            )
            self.current_goal_index += 1
            self.goal_sent = False
            self.navigation_complete = False
            return
    
    def execute_goals(self):
        """Start executing the goal sequence"""
        if not self.goals:
            self.get_logger().error("No goals to execute!")
            return
            
        self.get_logger().info(f"Starting execution of {len(self.goals)} goals...")
        
        # Initialize mission tracking
        self.mission_start_time = time.time()
        self.trajectory_recording_active = True
        self.get_logger().info(f"🔴 Trajectory recording activated. Position received: {self.position_received}")
        
        # Record initial position if we already have position data
        if self.position_received:
            self.record_trajectory_point()
            self.get_logger().info("Recorded initial trajectory point")
        else:
            self.get_logger().warn("No position data available yet for initial trajectory point")
        
        # Save goals to CSV before starting
        self.save_goals_to_csv()
        
        # Wait a moment for connections to establish
        time.sleep(1.0)
        
        # Start the goal sequence
        self.send_current_goal()
    
def main(args=None):
    rclpy.init(args=args)
    
    # Create the goal sender node
    goal_sender = SequentialGoalSender()
    
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 sequential_goal_sender.py <x1> <y1> [yaw1] <x2> <y2> [yaw2] ...")
        print("  python3 sequential_goal_sender.py --preset <preset_name>")
        print("")
        print("Examples:")
        print("  python3 sequential_goal_sender.py 10 5 20 10 30 15")
        print("  python3 sequential_goal_sender.py 10 5 1.57 20 10 0 30 15 -1.57")
        print("  python3 sequential_goal_sender.py --preset square")
        print("  python3 sequential_goal_sender.py --preset figure8")
        print("  python3 sequential_goal_sender.py --preset survey")
        print("  python3 sequential_goal_sender.py --preset astar_wp")
        print("  python3 sequential_goal_sender.py --preset astar_wp_file waypoints.csv")
        return
    
    # Handle preset patterns
    if sys.argv[1] == '--preset':
        if len(sys.argv) < 3:
            print("Please specify a preset name")
            return
            
        preset = sys.argv[2].lower()
        
        if preset == 'square':
            # Square pattern
            goal_sender.add_goal(10, 0, 0)
            goal_sender.add_goal(10, 10, math.pi/2)
            goal_sender.add_goal(0, 10, math.pi)
            goal_sender.add_goal(0, 0, -math.pi/2)
            
        elif preset == 'figure8':
            # Figure-8 pattern
            goal_sender.add_goal(10, 0, 0)
            goal_sender.add_goal(20, 10, math.pi/4)
            goal_sender.add_goal(10, 20, math.pi/2)
            goal_sender.add_goal(0, 10, -math.pi/4)
            goal_sender.add_goal(10, 0, 0)
            goal_sender.add_goal(20, -10, -math.pi/4)
            goal_sender.add_goal(10, -20, -math.pi/2)
            goal_sender.add_goal(0, -10, math.pi/4)
            
        elif preset == 'line':
            # Simple line pattern
            goal_sender.add_goal(10, 0, 0)
            goal_sender.add_goal(20, 0, 0)
            goal_sender.add_goal(30, 0, 0)
            goal_sender.add_goal(40, 0, 0)
            
        elif preset == 'zigzag':
            # Zigzag pattern
            goal_sender.add_goal(10, 0, 0)
            goal_sender.add_goal(20, 10, 0)
            goal_sender.add_goal(30, 0, 0)
            goal_sender.add_goal(40, 10, 0)
            goal_sender.add_goal(50, 0, 0)
            
        elif preset == 'survey':
            # Survey pattern - lawn mower style
            # Survey pattern east-west lines (replace previous lines)
            # Survey pattern east-west lines (lawn mower style)
            # Line 1: → (east)
            goal_sender.add_goal(-440, 202.0, 0)
            goal_sender.add_goal(-420, 202.0, 0)
            # Line 2: ← (west)
            goal_sender.add_goal(-420, 206.0, math.pi)
            goal_sender.add_goal(-440, 206.0, math.pi)
            # # Line 3: → (east)
            goal_sender.add_goal(-440, 210.0, 0)
            goal_sender.add_goal(-420, 210.0, 0)
            # # Line 4: ← (west)
            goal_sender.add_goal(-420, 214.0, math.pi)
            goal_sender.add_goal(-440, 214.0, math.pi)
            # # Line 5: → (east)
            goal_sender.add_goal(-440, 218.0, 0)
            goal_sender.add_goal(-420, 218.0, 0)
            # # Line 6: ← (west)
            goal_sender.add_goal(-420, 222.0, math.pi)
            goal_sender.add_goal(-440, 222.0, math.pi)
            # # Line 7: → (east)
            goal_sender.add_goal(-440, 226.0, 0)
            goal_sender.add_goal(-420, 226.0, 0)
            # # Line 8: ← (west)
            goal_sender.add_goal(-420, 230.0, math.pi)
            goal_sender.add_goal(-440, 230.0, math.pi)
            # # Line 9: → (east)
            goal_sender.add_goal(-440, 234.0, 0)
            goal_sender.add_goal(-420, 234.0, 0)
            # # Line 10: ← (west)
            goal_sender.add_goal(-420, 238.0, math.pi)
            goal_sender.add_goal(-440, 238.0, math.pi)
        
        elif preset == 'rrt':
            goal_sender.add_goal(-400, 200, 0)
            goal_sender.add_goal(-380.000717, 200.169372, 0)
            goal_sender.add_goal(-360.307233, 203.657446, 0)
            goal_sender.add_goal(-340.359067, 205.096438, 0)
            goal_sender.add_goal(-320.365058, 205.585915, 0)
            goal_sender.add_goal(-300.398078, 206.734697, 0)
            goal_sender.add_goal(-280.404138, 207.227027, 0)
            goal_sender.add_goal(-261.856356, 214.708991, 0)
            goal_sender.add_goal(-242.308655, 218.938341, 0)
            goal_sender.add_goal(-222.309269, 219.09495, 0)
            goal_sender.add_goal(-202.376202, 220.729838, 0)
            goal_sender.add_goal(-182.381515, 221.190783, 0)
            goal_sender.add_goal(-162.423912, 222.492358, 0)
            goal_sender.add_goal(-142.449891, 223.511413, 0)
            goal_sender.add_goal(-122.537973, 225.386385, 0)
            goal_sender.add_goal(-103.62933, 231.902767, 0)
            goal_sender.add_goal(-102.124445, 251.84607, 0)
            goal_sender.add_goal(-100.619559, 271.789373, 0)
            goal_sender.add_goal(-100, 280, 0)
        elif preset == 'vf-rrt':
            goal_sender.add_goal(-400, 200, 0)
            goal_sender.add_goal(-362.385986, 189.459206, 0)
            goal_sender.add_goal(-325.13309, 175.297859, 0)
            goal_sender.add_goal(-305.170102, 174.081661, 0)
            goal_sender.add_goal(-285.78416, 178.99951, 0)
            goal_sender.add_goal(-270.088675, 191.395144, 0)
            goal_sender.add_goal(-251.347738, 198.379218, 0)
            goal_sender.add_goal(-231.402639, 199.860107, 0)
            goal_sender.add_goal(-233.386675, 172.740811, 0)
            goal_sender.add_goal(-214.423782, 179.097595, 0)
            goal_sender.add_goal(-191.448336, 197.949809, 0)
            goal_sender.add_goal(-175.873038, 210.496127, 0)
            goal_sender.add_goal(-155.892334, 211.374468, 0)
            goal_sender.add_goal(-141.156082, 224.896408, 0)
            goal_sender.add_goal(-120.036463, 245.618272, 0)
            goal_sender.add_goal(-103.813898, 267.820295, 0)

        elif preset == 'svf-rrt':
            goal_sender.add_goal(-400, 200, 0)
            goal_sender.add_goal(-363.16926, 164.272295, 0)
            goal_sender.add_goal(-329.234613, 175.844342, 0)
            goal_sender.add_goal(-309.256508, 176.779926, 0)
            goal_sender.add_goal(-289.614682, 180.548041, 0)
            goal_sender.add_goal(-273.323747, 168.946083, 0)
            goal_sender.add_goal(-240.831318, 158.458674, 0)
            goal_sender.add_goal(-217.716956, 167.382973, 0)
            goal_sender.add_goal(-200.258906, 157.625079, 0)
            goal_sender.add_goal(-180.419931, 160.157875, 0)
            goal_sender.add_goal(-155.939546, 177.602314, 0)
            goal_sender.add_goal(-137.032717, 171.080672, 0)
            goal_sender.add_goal(-140.163521, 197.541973, 0)
            goal_sender.add_goal(-138.844136, 217.498406, 0)
            goal_sender.add_goal(-118.605342, 242.336565, 0)
            goal_sender.add_goal(-117.170739, 275.711898, 0)
        
        elif preset == 'survey_90':
            # Survey pattern rotated 90 degrees (north-south lines)
            goal_sender.add_goal(-438.0, 200.0, math.pi/2)   # Line 1 start: ↑ (north)
            goal_sender.add_goal(-438.0, 240.0, math.pi/2)   # Line 1 end: ↑ (north)
            goal_sender.add_goal(-434.0, 240.0, -math.pi/2)  # Line 2 start: ↓ (south)
            goal_sender.add_goal(-434.0, 200.0, -math.pi/2)  # Line 2 end: ↓ (south)
            goal_sender.add_goal(-430.0, 200.0, math.pi/2)   # Line 3 start: ↑ (north)
            goal_sender.add_goal(-430.0, 240.0, math.pi/2)   # Line 3 end: ↑ (north)
            goal_sender.add_goal(-426.0, 240.0, -math.pi/2)  # Line 4 start: ↓ (south)
            goal_sender.add_goal(-426.0, 200.0, -math.pi/2)  # Line 4 end: ↓ (south)
            goal_sender.add_goal(-422.0, 200.0, math.pi/2)   # Line 5 start: ↑ (north)
            goal_sender.add_goal(-422.0, 240.0, math.pi/2)   # Line 5 end: ↑ (north)
        elif preset == 'survey_45':
            goal_sender.add_goal(-425.0, 200.0, math.pi/4)
            goal_sender.add_goal(-420.0, 205.0, math.pi/4)

            goal_sender.add_goal(-420.0, 210.0, -3*math.pi/4)
            goal_sender.add_goal(-430.0, 200.0, -3*math.pi/4)

            goal_sender.add_goal(-435.0, 200.0, math.pi/4)
            goal_sender.add_goal(-420.0, 215.0, math.pi/4)

            goal_sender.add_goal(-420.0, 220.0, -3*math.pi/4)
            goal_sender.add_goal(-440.0, 200.0, -3*math.pi/4)

            goal_sender.add_goal(-440.0, 205.0, math.pi/4)
            goal_sender.add_goal(-420.0, 225.0, math.pi/4)

            goal_sender.add_goal(-420.0, 230.0, -3*math.pi/4)
            goal_sender.add_goal(-440.0, 210.0, -3*math.pi/4)

            goal_sender.add_goal(-440.0, 215.0, math.pi/4)
            goal_sender.add_goal(-420.0, 235.0, math.pi/4)

            goal_sender.add_goal(-420.0, 240.0, -3*math.pi/4)
            goal_sender.add_goal(-440.0, 220.0, -3*math.pi/4)

            goal_sender.add_goal(-440.0, 225.0, math.pi/4)
            goal_sender.add_goal(-425.0, 240.0, math.pi/4)

            goal_sender.add_goal(-430.0, 240.0, -3*math.pi/4)
            goal_sender.add_goal(-440.0, 230.0, -3*math.pi/4)

            goal_sender.add_goal(-440.0, 235.0, math.pi/4)
            goal_sender.add_goal(-435.0, 240.0, math.pi/4)
        elif preset == 'astar_wp':
            # Load waypoints from A* exported CSV file
            goal_sender.load_astar_waypoints_from_csv()
        elif preset == 'astar_wp':
            # Load waypoints from A* exported CSV file
            goal_sender.load_astar_waypoints_from_csv()
        elif preset == '2robots_1':
            # Load waypoints from A* exported CSV file
            goal_sender.load_astar_waypoints_from_csv(csv_file="2robots_1.csv")

        elif preset == 'astar_wp_file':
            # Load waypoints from specific A* CSV file (provided as third argument)
            if len(sys.argv) >= 4:
                csv_file = sys.argv[3]
                goal_sender.load_astar_waypoints_from_csv(csv_file)
            else:
                print("Please provide CSV filename: --preset astar_wp_file <filename.csv>")
                return
        
        else:
            print(f"Unknown preset: {preset}")
            print("Available presets: square, figure8, line, zigzag, survey, survey_90, astar_wp, astar_wp_file")
            return
    
    else:
        # Parse command line arguments for custom goals
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            if i + 1 < len(args):  # Need at least x and y
                x = float(args[i])
                y = float(args[i + 1])
                
                # Check if next argument is yaw or next x coordinate
                yaw = 0.0
                if (i + 2 < len(args) and 
                    (i + 3 >= len(args) or  # This is the last goal
                     not (i + 4 < len(args)))):  # Or next goal doesn't have enough args
                    try:
                        yaw = float(args[i + 2])
                        i += 3
                    except ValueError:
                        i += 2
                else:
                    i += 2
                
                goal_sender.add_goal(x, y, yaw)
            else:
                break
    
    if not goal_sender.goals:
        goal_sender.get_logger().error("No valid goals provided!")
        return
    
    # Start execution
    goal_sender.execute_goals()
    
    try:
        rclpy.spin(goal_sender)
    except KeyboardInterrupt:
        goal_sender.get_logger().info("Goal execution interrupted by user")
        # Stop trajectory recording
        goal_sender.trajectory_recording_active = False
        
        # Save both execution log and robot trajectory even if interrupted
        if goal_sender.execution_log:
            goal_sender.save_execution_log_to_csv()
        if goal_sender.trajectory_log:
            goal_sender.save_trajectory_to_csv()
            goal_sender.get_logger().info(f"Interrupted mission saved {len(goal_sender.trajectory_log)} trajectory points")
        else:
            goal_sender.get_logger().warn("No trajectory data to save - check if odometry topic is available")
    except Exception as e:
        goal_sender.get_logger().error(f"Error during execution: {e}")
    finally:
        # Clean shutdown
        try:
            goal_sender.destroy_node()
        except:
            pass
        
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
