#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <cmath>
#include <vector>
#include <limits>

class SimpleDWAPlanner : public rclcpp::Node
{
public:
  SimpleDWAPlanner() 
    : Node("simple_dwa_planner")
    , tf_buffer_(this->get_clock())
    , tf_listener_(tf_buffer_)
  {
    // Declare parameters
    this->declare_parameter("robot_frame", "wamv_simple");
    this->declare_parameter("global_frame", "odom");
    this->declare_parameter("control_frequency", 10.0);
    this->declare_parameter("goal_tolerance", 0.5);
    this->declare_parameter("max_vel_x", 5.0);
    this->declare_parameter("max_vel_theta", 2.0);
    this->declare_parameter("sim_time", 3.0);

    // Get parameters
    robot_frame_ = this->get_parameter("robot_frame").as_string();
    global_frame_ = this->get_parameter("global_frame").as_string();
    control_frequency_ = this->get_parameter("control_frequency").as_double();
    goal_tolerance_ = this->get_parameter("goal_tolerance").as_double();

    // Configure DWA
    config_.max_vel_x = this->get_parameter("max_vel_x").as_double();
    config_.max_vel_theta = this->get_parameter("max_vel_theta").as_double();
    config_.min_vel_theta = -config_.max_vel_theta;
    config_.sim_time = this->get_parameter("sim_time").as_double();
    
    // Publishers
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/wamv/cmd_vel", 1);
    status_pub_ = this->create_publisher<std_msgs::msg::String>("navigation_status", 1);
    trajectory_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("dwa_trajectories", 1);
    
    // Subscribers
    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "goal_pose", 1,
      std::bind(&SimpleDWAPlanner::goalCallback, this, std::placeholders::_1));
    
    // Remove odometry subscriber - we'll use TF transforms instead
    
        // Create timer for control loop (50 Hz for very fast response)
    control_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(20),  // 20ms = 50Hz (was 50ms = 20Hz)
      [this]() { this->controlCallback(); }
    );
    
    RCLCPP_INFO(this->get_logger(), "Simple DWA Planner initialized");
    publishStatus("IDLE: Waiting for goal");
  }

private:
  struct DWAConfig
  {
    double max_vel_x = 50.0;           // High max velocity
    double min_vel_x = 2.0;            // Higher minimum velocity
    double max_vel_theta = 5.0;        // Fast rotation
    double min_vel_theta = -5.0;       // Fast rotation
    double acc_lim_x = 10.0;           // Very fast acceleration
    double acc_lim_theta = 8.0;        // Very fast angular acceleration
    double sim_time = 1.0;             // Short prediction time
    double sim_granularity = 0.2;      // Coarser granularity for speed
    double goal_distance_bias = 5.0;   // Strong bias toward goal (was 2.0)
    double heading_bias = 0.5;         // Reduced heading bias for faster turns (was 1.0)
    double velocity_bias = 2.0;        // Strong preference for high velocity (was 0.1)
    int vel_samples = 5;               // Fewer samples for speed
    int angular_samples = 10;          // Fewer samples for speed
  };
  
  struct Trajectory
  {
    std::vector<geometry_msgs::msg::PoseStamped> poses;
    double linear_vel;
    double angular_vel;
    double cost;
    
    Trajectory() : linear_vel(0.0), angular_vel(0.0), cost(std::numeric_limits<double>::max()) {}
  };
  
  void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    goal_pose_ = *msg;
    has_goal_ = true;
    RCLCPP_INFO(this->get_logger(), "Received goal: (%.2f, %.2f)", 
                msg->pose.position.x, msg->pose.position.y);
    publishStatus("NAVIGATING: Moving to goal");
  }
  
  bool getRobotPose(geometry_msgs::msg::PoseStamped& robot_pose)
  {
    try {
      auto transform = tf_buffer_.lookupTransform(
        global_frame_, robot_frame_, tf2::TimePointZero);
      
      robot_pose.header.frame_id = global_frame_;
      robot_pose.header.stamp = this->now();
      robot_pose.pose.position.x = transform.transform.translation.x;
      robot_pose.pose.position.y = transform.transform.translation.y;
      robot_pose.pose.position.z = transform.transform.translation.z;
      robot_pose.pose.orientation = transform.transform.rotation;

      
      return true;
    } catch (tf2::TransformException& ex) {
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                           "Could not get robot pose: %s", ex.what());
      return false;
    }
  }
  
  void controlCallback()
  {
    if (!has_goal_) {
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "No goal set");
      return;
    }
    
    // Get current robot pose from TF
    if (!getRobotPose(current_pose_)) {
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Could not get robot pose from TF");
      return;
    }
    
    // For velocity, we'll assume zero for now (could estimate from pose changes)
    current_vel_.linear.x = 0.0;
    current_vel_.angular.z = 0.0;
    
    // Check if goal is reached
    double distance_to_goal = calculateDistance(current_pose_, goal_pose_);
    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                         "Distance to goal: %.3f, robot at (%.2f, %.2f)", 
                         distance_to_goal, current_pose_.pose.position.x, current_pose_.pose.position.y);
    
    if (distance_to_goal < goal_tolerance_) {
      geometry_msgs::msg::Twist stop_cmd;
      cmd_vel_pub_->publish(stop_cmd);
      publishStatus("GOAL_REACHED: Target reached successfully");
      RCLCPP_INFO(this->get_logger(), "Goal reached!");
      has_goal_ = false;
      return;
    }
    
    // Compute velocity command using DWA
    geometry_msgs::msg::Twist cmd_vel;
    if (computeVelocityCommand(cmd_vel)) {
      cmd_vel_pub_->publish(cmd_vel);
      publishTrajectoryMarkers();
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                          "Published cmd_vel: linear=%.2f, angular=%.2f", 
                          cmd_vel.linear.x, cmd_vel.angular.z);
    } else {
      geometry_msgs::msg::Twist stop_cmd;
      cmd_vel_pub_->publish(stop_cmd);
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "DWA failed to find valid command");
    }
  }
  
  bool computeVelocityCommand(geometry_msgs::msg::Twist& cmd_vel)
  {
    all_trajectories_.clear();
    
    double best_cost = std::numeric_limits<double>::max();
    geometry_msgs::msg::Twist best_cmd_vel;
    
    // Calculate dynamic window based on current velocity
    double dt = config_.sim_granularity;
    double min_vel = std::max(config_.min_vel_x, current_vel_.linear.x - config_.acc_lim_x * dt);
    double max_vel = std::min(config_.max_vel_x, current_vel_.linear.x + config_.acc_lim_x * dt);
    double min_yaw_rate = std::max(config_.min_vel_theta, current_vel_.angular.z - config_.acc_lim_theta * dt);
    double max_yaw_rate = std::min(config_.max_vel_theta, current_vel_.angular.z + config_.acc_lim_theta * dt);

    // Sample velocities within dynamic window
    double vel_step = (max_vel - min_vel) / config_.vel_samples;
    double yaw_rate_step = (max_yaw_rate - min_yaw_rate) / config_.angular_samples;
    
    for (int i = 0; i <= config_.vel_samples; ++i) {
      double sample_vel = min_vel + i * vel_step;
      
      for (int j = 0; j <= config_.angular_samples; ++j) {
        double sample_yaw_rate = min_yaw_rate + j * yaw_rate_step;
        
        // Generate trajectory
        Trajectory trajectory = generateTrajectory(sample_vel, sample_yaw_rate);
        
        if (trajectory.poses.size() < 2) {
          continue;
        }
        
        // Score trajectory
        double cost = scoreTrajectory(trajectory);
        trajectory.cost = cost;
        all_trajectories_.push_back(trajectory);
        
        // Update best trajectory
        if (cost < best_cost) {
          best_cost = cost;
          best_cmd_vel.linear.x = sample_vel;
          best_cmd_vel.angular.z = -sample_yaw_rate;  // Reverse rotation direction
          best_trajectory_ = trajectory;
        }
      }
    }
    
    cmd_vel = best_cmd_vel;
    return best_cost < std::numeric_limits<double>::max();
  }
  
  Trajectory generateTrajectory(double sample_vel, double sample_yaw_rate)
  {
    Trajectory trajectory;
    trajectory.linear_vel = sample_vel;
    trajectory.angular_vel = sample_yaw_rate;
    
    // Start from current pose
    geometry_msgs::msg::PoseStamped pose = current_pose_;
    double yaw = getYaw(pose.pose.orientation);
    
    // Simulate trajectory
    double time = 0.0;
    while (time <= config_.sim_time) {
      trajectory.poses.push_back(pose);
      
      // Update pose based on velocity
      double dt = config_.sim_granularity;
      pose.pose.position.x += sample_vel * cos(yaw) * dt;
      pose.pose.position.y += sample_vel * sin(yaw) * dt;
      yaw += sample_yaw_rate * dt;
      yaw = normalizeAngle(yaw);
      
      // Update orientation
      pose.pose.orientation = createQuaternionFromYaw(yaw);
      
      time += dt;
    }
    
    return trajectory;
  }
  
  double scoreTrajectory(const Trajectory& trajectory)
  {
    double goal_cost = calculateGoalCost(trajectory);
    double heading_cost = calculateHeadingCost(trajectory);
    double velocity_cost = calculateVelocityCost(trajectory);
    
    return config_.goal_distance_bias * goal_cost +
           config_.heading_bias * heading_cost +
           config_.velocity_bias * velocity_cost;
  }
  
  double calculateGoalCost(const Trajectory& trajectory)
  {
    if (trajectory.poses.empty()) {
      return 1000.0;
    }
    return calculateDistance(trajectory.poses.back(), goal_pose_);
  }
  
  double calculateHeadingCost(const Trajectory& trajectory)
  {
    if (trajectory.poses.size() < 2) {
      return 0.0;
    }
    
    // Get trajectory heading
    const auto& start_pose = trajectory.poses.front();
    const auto& end_pose = trajectory.poses.back();
    double traj_heading = atan2(
      end_pose.pose.position.y - start_pose.pose.position.y,
      end_pose.pose.position.x - start_pose.pose.position.x
    );
    
    // Get desired heading to goal
    double desired_heading = atan2(
      goal_pose_.pose.position.y - start_pose.pose.position.y,
      goal_pose_.pose.position.x - start_pose.pose.position.x
    );
    
    double heading_error = fabs(normalizeAngle(traj_heading - desired_heading));
    return heading_error;
  }
  
  double calculateVelocityCost(const Trajectory& trajectory)
  {
    return config_.max_vel_x - trajectory.linear_vel;
  }
  
  double getYaw(const geometry_msgs::msg::Quaternion& q)
  {
    // Convert quaternion to yaw
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    return atan2(siny_cosp, cosy_cosp);
  }
  
  geometry_msgs::msg::Quaternion createQuaternionFromYaw(double yaw)
  {
    geometry_msgs::msg::Quaternion q;
    q.x = 0.0;
    q.y = 0.0;
    q.z = sin(yaw / 2.0);
    q.w = cos(yaw / 2.0);
    return q;
  }
  
  double normalizeAngle(double angle)
  {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
  }
  
  double calculateDistance(
    const geometry_msgs::msg::PoseStamped& pose1,
    const geometry_msgs::msg::PoseStamped& pose2)
  {
    double dx = pose1.pose.position.x - pose2.pose.position.x;
    double dy = pose1.pose.position.y - pose2.pose.position.y;
    return sqrt(dx * dx + dy * dy);
  }
  
  void publishStatus(const std::string& status)
  {
    std_msgs::msg::String msg;
    msg.data = status;
    status_pub_->publish(msg);
  }
  
  void publishTrajectoryMarkers()
  {
    visualization_msgs::msg::MarkerArray marker_array;
    
    for (size_t i = 0; i < all_trajectories_.size(); ++i) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = global_frame_;
      marker.header.stamp = this->now();
      marker.ns = "dwa_trajectories";
      marker.id = static_cast<int>(i);
      marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.scale.x = 2.0;  // Increased 100x from 0.02 to 2.0
      
      // Color based on trajectory quality
      if (&all_trajectories_[i] == &best_trajectory_) {
        // Best trajectory in green
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        marker.scale.x = 5.0;  // Increased 100x from 0.05 to 5.0
      } else {
        // Other trajectories in blue
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 0.3;
      }
      
      // Add trajectory points
      for (const auto& pose : all_trajectories_[i].poses) {
        geometry_msgs::msg::Point point;
        point.x = pose.pose.position.x;
        point.y = pose.pose.position.y;
        point.z = pose.pose.position.z;
        marker.points.push_back(point);
      }
      
      marker_array.markers.push_back(marker);
    }
    
    trajectory_markers_pub_->publish(marker_array);
  }
  
  // Publishers and subscribers
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trajectory_markers_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  
  // Timer
  rclcpp::TimerBase::SharedPtr control_timer_;
  
  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  
  // State
  geometry_msgs::msg::PoseStamped current_pose_;
  geometry_msgs::msg::PoseStamped goal_pose_;
  geometry_msgs::msg::Twist current_vel_;

  bool has_goal_ = false;
  
  // DWA
  DWAConfig config_;
  Trajectory best_trajectory_;
  std::vector<Trajectory> all_trajectories_;
  
  // Parameters
  std::string robot_frame_;
  std::string global_frame_;
  double control_frequency_;
  double goal_tolerance_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SimpleDWAPlanner>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
