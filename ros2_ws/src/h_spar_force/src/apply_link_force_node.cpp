#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include <gz/transport/Node.hh>
#include <gz/msgs/entity.pb.h>
#include <gz/msgs/entity_wrench.pb.h>

using namespace std::chrono_literals;

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

    world_name_ = this->get_parameter("world_name").as_string();
    link_name_ = this->get_parameter("link_name").as_string();
    ros_topic_ = this->get_parameter("ros_topic").as_string();
    persistent_ = this->get_parameter("persistent").as_bool();
    command_timeout_ = this->get_parameter("command_timeout").as_double();
    zero_tolerance_ = this->get_parameter("zero_tolerance").as_double();

    if (persistent_)
    {
      gz_wrench_topic_ = "/world/" + world_name_ + "/wrench/persistent";
    }
    else
    {
      gz_wrench_topic_ = "/world/" + world_name_ + "/wrench";
    }

    gz_clear_topic_ = "/world/" + world_name_ + "/wrench/clear";

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

    RCLCPP_INFO(this->get_logger(), "ROS topic: %s", ros_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Gazebo wrench topic: %s", gz_wrench_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Gazebo clear topic: %s", gz_clear_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Target link: %s", link_name_.c_str());

    subscription_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
      ros_topic_,
      rclcpp::QoS(10),
      std::bind(&RosToGazeboWrenchBridge::wrench_callback, this, std::placeholders::_1));

    last_command_time_ = this->now();
    command_active_ = false;
    clear_sent_ = true;

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
    last_command_time_ = this->now();

    if (is_zero_wrench(ros_msg))
    {
      clear_wrench();
      command_active_ = false;
      clear_sent_ = true;

      RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "Received zero wrench. Cleared persistent Gazebo wrench.");

      return;
    }

    publish_wrench(
      ros_msg->wrench.force.x,
      ros_msg->wrench.force.y,
      ros_msg->wrench.force.z,
      ros_msg->wrench.torque.x,
      ros_msg->wrench.torque.y,
      ros_msg->wrench.torque.z);

    command_active_ = true;
    clear_sent_ = false;
  }

  void watchdog_callback()
  {
    if (!command_active_)
    {
      return;
    }

    const double elapsed = (this->now() - last_command_time_).seconds();

    if (elapsed > command_timeout_ && !clear_sent_)
    {
      clear_wrench();

      RCLCPP_WARN(
        this->get_logger(),
        "No wrench command received for %.3f s. Cleared persistent Gazebo wrench.",
        elapsed);

      clear_sent_ = true;
      command_active_ = false;
    }
  }

  void publish_wrench(
    double fx, double fy, double fz,
    double tx, double ty, double tz)
  {
    gz::msgs::EntityWrench gz_msg;

    gz_msg.mutable_entity()->set_name(link_name_);
    gz_msg.mutable_entity()->set_type(gz::msgs::Entity::LINK);

    gz_msg.mutable_wrench()->mutable_force()->set_x(fx);
    gz_msg.mutable_wrench()->mutable_force()->set_y(fy);
    gz_msg.mutable_wrench()->mutable_force()->set_z(fz);

    gz_msg.mutable_wrench()->mutable_torque()->set_x(tx);
    gz_msg.mutable_wrench()->mutable_torque()->set_y(ty);
    gz_msg.mutable_wrench()->mutable_torque()->set_z(tz);

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

  bool persistent_;
  bool command_active_;
  bool clear_sent_;

  double command_timeout_;
  double zero_tolerance_;

  rclcpp::Time last_command_time_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RosToGazeboWrenchBridge>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}