#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include <gz/transport/Node.hh>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/entity.pb.h>
#include <gz/msgs/entity_wrench.pb.h>

using namespace std::chrono_literals;

// Gazebo's ApplyLinkWrench system keeps every message published on
// /world/<world>/wrench/persistent and applies their sum each step until a
// clear. So in persistent mode each command is published as the difference
// from what Gazebo already holds, which makes the sum equal the latest command
// regardless of the real-time factor.
//
// Gazebo processes a clear on its next step but stores new wrenches
// immediately, so anything published between a clear and that step is erased
// too. After a clear the bridge therefore waits until simulation time has
// advanced by clear_wait before publishing again. Clears happen on a zero
// command, on the watchdog timeout, and every rebase_count commands to keep
// Gazebo's list of wrenches short.
class RosToGazeboWrenchBridge : public rclcpp::Node
{
public:
  RosToGazeboWrenchBridge()
  : Node("ros_to_gazebo_wrench_bridge")
  {
    this->declare_parameter<std::string>("world_name", "sydney_regatta");
    this->declare_parameter<std::string>("link_name", "wamv/base_link");
    this->declare_parameter<std::string>("ros_topic", "/h_spar/wamv/wrench_cmd");

    this->declare_parameter<bool>("persistent", true);
    this->declare_parameter<double>("command_timeout", 0.5);
    this->declare_parameter<double>("watchdog_rate", 20.0);
    this->declare_parameter<double>("zero_tolerance", 1e-9);
    // Simulation time to wait after a clear before publishing again (s)
    this->declare_parameter<double>("clear_wait", 0.1);
    // Number of difference commands Gazebo holds before the bridge clears and
    // republishes the full wrench
    this->declare_parameter<int>("rebase_count", 300);

    world_name_ = this->get_parameter("world_name").as_string();
    link_name_ = this->get_parameter("link_name").as_string();
    ros_topic_ = this->get_parameter("ros_topic").as_string();
    persistent_ = this->get_parameter("persistent").as_bool();
    command_timeout_ = this->get_parameter("command_timeout").as_double();
    zero_tolerance_ = this->get_parameter("zero_tolerance").as_double();
    clear_wait_ = this->get_parameter("clear_wait").as_double();
    rebase_count_ = this->get_parameter("rebase_count").as_int();

    if (persistent_)
    {
      gz_wrench_topic_ = "/world/" + world_name_ + "/wrench/persistent";
    }
    else
    {
      gz_wrench_topic_ = "/world/" + world_name_ + "/wrench";
    }

    gz_clear_topic_ = "/world/" + world_name_ + "/wrench/clear";
    gz_clock_topic_ = "/world/" + world_name_ + "/clock";

    gz_wrench_pub_ = gz_node_.Advertise<gz::msgs::EntityWrench>(gz_wrench_topic_);
    gz_clear_pub_ = gz_node_.Advertise<gz::msgs::Entity>(gz_clear_topic_);

    if (!gz_wrench_pub_)
    {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to advertise Gazebo wrench topic: %s",
        gz_wrench_topic_.c_str());
    }

    if (!gz_clear_pub_)
    {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to advertise Gazebo clear topic: %s",
        gz_clear_topic_.c_str());
    }

    if (!gz_node_.Subscribe(gz_clock_topic_, &RosToGazeboWrenchBridge::clock_callback, this))
    {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to subscribe to Gazebo clock topic: %s",
        gz_clock_topic_.c_str());
    }

    RCLCPP_INFO(this->get_logger(), "ROS topic: %s", ros_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Gazebo wrench topic: %s", gz_wrench_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Gazebo clear topic: %s", gz_clear_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Gazebo clock topic: %s", gz_clock_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Target link: %s", link_name_.c_str());

    subscription_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
      ros_topic_,
      rclcpp::QoS(10),
      std::bind(&RosToGazeboWrenchBridge::wrench_callback, this, std::placeholders::_1));

    last_command_time_ = this->now();
    command_active_ = false;

    double watchdog_rate = this->get_parameter("watchdog_rate").as_double();
    auto watchdog_period = std::chrono::duration<double>(1.0 / watchdog_rate);

    watchdog_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(watchdog_period),
      std::bind(&RosToGazeboWrenchBridge::watchdog_callback, this));
  }

  ~RosToGazeboWrenchBridge()
  {
    clear_wrench();
  }

private:
  bool is_zero_wrench(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) const
  {
    return std::abs(msg->wrench.force.x) <= zero_tolerance_ &&
           std::abs(msg->wrench.force.y) <= zero_tolerance_ &&
           std::abs(msg->wrench.force.z) <= zero_tolerance_ &&
           std::abs(msg->wrench.torque.x) <= zero_tolerance_ &&
           std::abs(msg->wrench.torque.y) <= zero_tolerance_ &&
           std::abs(msg->wrench.torque.z) <= zero_tolerance_;
  }

  void wrench_callback(const geometry_msgs::msg::WrenchStamped::SharedPtr ros_msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    last_command_time_ = this->now();

    if (!persistent_)
    {
      if (!is_zero_wrench(ros_msg))
      {
        publish_wrench(ros_msg->wrench);
      }
      return;
    }

    if (is_zero_wrench(ros_msg))
    {
      if (command_active_)
      {
        start_clear();
        command_active_ = false;

        RCLCPP_INFO_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          1000,
          "Received zero wrench. Cleared persistent Gazebo wrench.");
      }
      return;
    }

    target_ = ros_msg->wrench;
    command_active_ = true;

    if (waiting_for_clear_)
    {
      // The full target is published once the clear has been processed
      return;
    }

    // Gazebo may still hold wrenches from a previous bridge instance
    if (!cleared_once_ || held_count_ >= rebase_count_)
    {
      start_clear();
      return;
    }

    publish_difference();
  }

  void clock_callback(const gz::msgs::Clock & msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    sim_time_ = msg.sim().sec() + msg.sim().nsec() * 1e-9;
    have_sim_time_ = true;

    if (waiting_for_clear_ && clear_sim_time_ < 0.0)
    {
      // Clear was sent before the first clock message
      clear_sim_time_ = sim_time_;
    }

    if (waiting_for_clear_ && sim_time_ >= clear_sim_time_ + clear_wait_)
    {
      waiting_for_clear_ = false;
      if (command_active_)
      {
        publish_difference();
      }
    }
  }

  void watchdog_callback()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!command_active_)
    {
      return;
    }

    const double elapsed = (this->now() - last_command_time_).seconds();

    if (elapsed > command_timeout_)
    {
      start_clear();
      command_active_ = false;

      RCLCPP_WARN(
        this->get_logger(),
        "No wrench command received for %.3f s. Cleared persistent Gazebo wrench.",
        elapsed);
    }
  }

  void start_clear()
  {
    clear_wrench();
    cleared_once_ = true;
    held_ = geometry_msgs::msg::Wrench();
    held_count_ = 0;
    waiting_for_clear_ = true;
    clear_sim_time_ = have_sim_time_ ? sim_time_ : -1.0;
  }

  void publish_difference()
  {
    geometry_msgs::msg::Wrench delta;
    delta.force.x = target_.force.x - held_.force.x;
    delta.force.y = target_.force.y - held_.force.y;
    delta.force.z = target_.force.z - held_.force.z;
    delta.torque.x = target_.torque.x - held_.torque.x;
    delta.torque.y = target_.torque.y - held_.torque.y;
    delta.torque.z = target_.torque.z - held_.torque.z;

    publish_wrench(delta);
    held_ = target_;
    ++held_count_;
  }

  void publish_wrench(const geometry_msgs::msg::Wrench & wrench)
  {
    gz::msgs::EntityWrench gz_msg;

    gz_msg.mutable_entity()->set_name(link_name_);
    gz_msg.mutable_entity()->set_type(gz::msgs::Entity::LINK);

    gz_msg.mutable_wrench()->mutable_force()->set_x(wrench.force.x);
    gz_msg.mutable_wrench()->mutable_force()->set_y(wrench.force.y);
    gz_msg.mutable_wrench()->mutable_force()->set_z(wrench.force.z);

    gz_msg.mutable_wrench()->mutable_torque()->set_x(wrench.torque.x);
    gz_msg.mutable_wrench()->mutable_torque()->set_y(wrench.torque.y);
    gz_msg.mutable_wrench()->mutable_torque()->set_z(wrench.torque.z);

    gz_wrench_pub_.Publish(gz_msg);
  }

  void clear_wrench()
  {
    gz::msgs::Entity clear_msg;
    clear_msg.set_name(link_name_);
    clear_msg.set_type(gz::msgs::Entity::LINK);

    gz_clear_pub_.Publish(clear_msg);
  }

private:
  gz::transport::Node gz_node_;
  gz::transport::Node::Publisher gz_wrench_pub_;
  gz::transport::Node::Publisher gz_clear_pub_;

  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;

  std::string world_name_;
  std::string link_name_;
  std::string ros_topic_;
  std::string gz_wrench_topic_;
  std::string gz_clear_topic_;
  std::string gz_clock_topic_;

  bool persistent_;
  bool command_active_;

  double command_timeout_;
  double zero_tolerance_;
  double clear_wait_;
  int64_t rebase_count_;

  rclcpp::Time last_command_time_;

  // Guards the state below, shared with the Gazebo clock callback thread
  std::mutex mutex_;
  geometry_msgs::msg::Wrench target_;  // latest command
  geometry_msgs::msg::Wrench held_;    // sum of wrenches Gazebo currently holds
  int64_t held_count_ = 0;
  bool waiting_for_clear_ = false;
  bool cleared_once_ = false;
  bool have_sim_time_ = false;
  double sim_time_ = 0.0;
  double clear_sim_time_ = -1.0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RosToGazeboWrenchBridge>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
