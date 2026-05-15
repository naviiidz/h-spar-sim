#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float64.hpp>

class USVVelocityController : public rclcpp::Node
{
public:
    USVVelocityController() : Node("usv_velocity_controller")
    {
        // Subscribe to cmd_vel topic
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/wamv/cmd_vel", 10,
            std::bind(&USVVelocityController::cmdVelCallback, this, std::placeholders::_1));
        
        // Publishers for VRX thrust commands
        left_thrust_pub_ = this->create_publisher<std_msgs::msg::Float64>("/wamv/thrusters/left/thrust", 10);
        right_thrust_pub_ = this->create_publisher<std_msgs::msg::Float64>("/wamv/thrusters/right/thrust", 10);
        
        // Parameters
        this->declare_parameter("max_thrust", 250.0);  // Maximum thrust in Newtons
        this->declare_parameter("boat_length", 4.9);   // Distance between thrusters (WAM-V length)
        this->declare_parameter("thrust_deadband", 0.1); // Minimum thrust to overcome friction
        
        max_thrust_ = this->get_parameter("max_thrust").as_double();
        boat_length_ = this->get_parameter("boat_length").as_double();
        thrust_deadband_ = this->get_parameter("thrust_deadband").as_double();
        
        RCLCPP_INFO(this->get_logger(), "USV Velocity Controller initialized");
        RCLCPP_INFO(this->get_logger(), "Max thrust: %.1f N", max_thrust_);
        RCLCPP_INFO(this->get_logger(), "Boat length: %.1f m", boat_length_);
        RCLCPP_INFO(this->get_logger(), "Listening for cmd_vel on: /wamv/cmd_vel");
    }

private:
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_thrust_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr right_thrust_pub_;
    
    double max_thrust_;
    double boat_length_;
    double thrust_deadband_;
    
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        // Extract desired velocities
        double linear_vel = msg->linear.x;   // Forward/backward velocity (m/s)
        double angular_vel = msg->angular.z; // Yaw rate (rad/s)
        
        // Convert to differential thrust commands
        // For a differential drive boat:
        // left_thrust + right_thrust = total_thrust (for linear motion)
        // left_thrust - right_thrust = differential_thrust (for angular motion)
        
        // Calculate thrust components
        double linear_thrust = linear_vel * 100.0;  // Scale factor for linear motion
        double angular_thrust = angular_vel * boat_length_ * 50.0; // Scale factor for angular motion
        
        // Calculate individual thruster commands
        double left_thrust = linear_thrust + angular_thrust;
        double right_thrust = linear_thrust - angular_thrust;
        
        // Apply thrust limits
        left_thrust = std::max(-max_thrust_, std::min(max_thrust_, left_thrust));
        right_thrust = std::max(-max_thrust_, std::min(max_thrust_, right_thrust));
        
        // Apply deadband to overcome friction
        if (std::abs(left_thrust) < thrust_deadband_) {
            left_thrust = 0.0;
        }
        if (std::abs(right_thrust) < thrust_deadband_) {
            right_thrust = 0.0;
        }
        
        // Publish thrust commands
        auto left_msg = std_msgs::msg::Float64();
        auto right_msg = std_msgs::msg::Float64();
        left_msg.data = left_thrust;
        right_msg.data = right_thrust;
        
        left_thrust_pub_->publish(left_msg);
        right_thrust_pub_->publish(right_msg);
        
        // Log the commands
        if (linear_vel != 0.0 || angular_vel != 0.0) {
            RCLCPP_DEBUG(this->get_logger(), 
                "Linear: %.2f m/s, Angular: %.2f rad/s -> Left: %.1f N, Right: %.1f N",
                linear_vel, angular_vel, left_thrust, right_thrust);
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<USVVelocityController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
