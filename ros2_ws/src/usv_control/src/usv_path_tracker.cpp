#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

class USVPathTracker : public rclcpp::Node
{
private:
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
    
    nav_msgs::msg::Path robot_path_;
    std::string map_frame_;
    double min_distance_; // Minimum distance to add new point

public:
    USVPathTracker() : Node("usv_path_tracker")
    {
        this->declare_parameter<std::string>("map_frame", "map");
        this->declare_parameter<double>("min_distance", 0.5); // meters
        
        this->get_parameter("map_frame", map_frame_);
        this->get_parameter("min_distance", min_distance_);
        
        // Initialize path
        robot_path_.header.frame_id = map_frame_;
        
        // Publisher and subscriber
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("usv_path", 10);
        pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/wamv/pose", 10,
            std::bind(&USVPathTracker::pose_callback, this, std::placeholders::_1));
            
        RCLCPP_INFO(this->get_logger(), "USV Path Tracker initialized");
    }

private:
    void pose_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = map_frame_;
        pose_stamped.header.stamp = this->get_clock()->now();
        pose_stamped.pose = msg->pose.pose;
        
        // Add to path if it's far enough from the last point
        if (should_add_point(pose_stamped.pose.position)) {
            robot_path_.poses.push_back(pose_stamped);
            robot_path_.header.stamp = this->get_clock()->now();
            
            // Keep path length reasonable (last 1000 points)
            if (robot_path_.poses.size() > 1000) {
                robot_path_.poses.erase(robot_path_.poses.begin());
            }
            
            path_pub_->publish(robot_path_);
        }
    }
    
    bool should_add_point(const geometry_msgs::msg::Point& new_point)
    {
        if (robot_path_.poses.empty()) {
            return true;
        }
        
        auto& last_point = robot_path_.poses.back().pose.position;
        double distance = sqrt(
            pow(new_point.x - last_point.x, 2) +
            pow(new_point.y - last_point.y, 2)
        );
        
        return distance >= min_distance_;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<USVPathTracker>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
