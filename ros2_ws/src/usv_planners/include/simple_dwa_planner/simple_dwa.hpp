#ifndef SIMPLE_DWA_PLANNER_SIMPLE_DWA_HPP_
#define SIMPLE_DWA_PLANNER_SIMPLE_DWA_HPP_

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

namespace simple_dwa_planner
{

struct DWAConfig
{
  // Robot constraints
  double max_vel_x = 2.0;
  double min_vel_x = 0.0;
  double max_vel_theta = 1.0;
  double min_vel_theta = -1.0;
  double acc_lim_x = 1.0;
  double acc_lim_theta = 1.0;
  
  // Simulation parameters
  double sim_time = 3.0;
  double sim_granularity = 0.1;
  
  // Scoring parameters
  double goal_distance_bias = 2.0;
  double heading_bias = 1.0;
  double velocity_bias = 0.1;
  
  // Velocity sampling
  int vel_samples = 10;
  int angular_samples = 20;
};

struct Trajectory
{
  std::vector<geometry_msgs::msg::PoseStamped> poses;
  double linear_vel;
  double angular_vel;
  double cost;
  
  Trajectory() : linear_vel(0.0), angular_vel(0.0), cost(std::numeric_limits<double>::max()) {}
};

class SimpleDWA
{
public:
  SimpleDWA();
  ~SimpleDWA();
  
  void configure(const DWAConfig& config);
  
  bool computeVelocityCommand(
    const geometry_msgs::msg::PoseStamped& current_pose,
    const geometry_msgs::msg::Twist& current_vel,
    const geometry_msgs::msg::PoseStamped& goal_pose,
    geometry_msgs::msg::Twist& cmd_vel
  );
  
  const std::vector<Trajectory>& getAllTrajectories() const { return all_trajectories_; }
  const Trajectory& getBestTrajectory() const { return best_trajectory_; }

private:
  Trajectory generateTrajectory(
    const geometry_msgs::msg::PoseStamped& current_pose,
    double sample_vel,
    double sample_yaw_rate
  );
  
  double scoreTrajectory(
    const Trajectory& trajectory,
    const geometry_msgs::msg::PoseStamped& goal_pose
  );
  
  double calculateGoalCost(
    const Trajectory& trajectory,
    const geometry_msgs::msg::PoseStamped& goal_pose
  );
  
  double calculateHeadingCost(
    const Trajectory& trajectory,
    const geometry_msgs::msg::PoseStamped& goal_pose
  );
  
  double calculateVelocityCost(const Trajectory& trajectory);
  
  double normalizeAngle(double angle);
  
  double calculateDistance(
    const geometry_msgs::msg::PoseStamped& pose1,
    const geometry_msgs::msg::PoseStamped& pose2
  );
  
  double getYaw(const geometry_msgs::msg::Quaternion& q);
  
  geometry_msgs::msg::Quaternion createQuaternionFromYaw(double yaw);
  
  DWAConfig config_;
  Trajectory best_trajectory_;
  std::vector<Trajectory> all_trajectories_;
};

}  // namespace simple_dwa_planner

#endif  // SIMPLE_DWA_PLANNER_SIMPLE_DWA_HPP_
