#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <termios.h>
#include <unistd.h>
#include <stdio.h>
#include <signal.h>

class USVTeleopKeyboard : public rclcpp::Node
{
public:
    USVTeleopKeyboard() : Node("usv_teleop_keyboard")
    {
        // Publisher for cmd_vel topic
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/wamv/cmd_vel", 10);
        
        // Parameters for velocity limits
        this->declare_parameter("linear_speed", 2.0);
        this->declare_parameter("angular_speed", 1.0);
        this->declare_parameter("linear_step", 0.1);
        this->declare_parameter("angular_step", 0.1);
        
        linear_speed_ = this->get_parameter("linear_speed").as_double();
        angular_speed_ = this->get_parameter("angular_speed").as_double();
        linear_step_ = this->get_parameter("linear_step").as_double();
        angular_step_ = this->get_parameter("angular_step").as_double();
        
        // Initialize velocities
        linear_vel_ = 0.0;
        angular_vel_ = 0.0;
        
        RCLCPP_INFO(this->get_logger(), "USV Teleop Keyboard Controller initialized");
        RCLCPP_INFO(this->get_logger(), "Max linear speed: %.1f m/s", linear_speed_);
        RCLCPP_INFO(this->get_logger(), "Max angular speed: %.1f rad/s", angular_speed_);
        printInstructions();
        
        // Set up terminal for raw input
        setupTerminal();
    }
    
    ~USVTeleopKeyboard()
    {
        restoreTerminal();
    }
    
    void run()
    {
        char key;
        while (rclcpp::ok()) {
            if (read(STDIN_FILENO, &key, 1) == 1) {
                processKey(key);
                publishVelocity();
            }
            rclcpp::spin_some(shared_from_this());
        }
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    
    double linear_speed_;
    double angular_speed_;
    double linear_step_;
    double angular_step_;
    double linear_vel_;
    double angular_vel_;
    
    struct termios old_terminal_settings_;
    
    void setupTerminal()
    {
        tcgetattr(STDIN_FILENO, &old_terminal_settings_);
        struct termios new_settings = old_terminal_settings_;
        new_settings.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &new_settings);
    }
    
    void restoreTerminal()
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_terminal_settings_);
    }
    
    void printInstructions()
    {
        printf("\n");
        printf("USV Teleop Keyboard Control\n");
        printf("===========================\n");
        printf("Use the following keys to control the USV:\n");
        printf("\n");
        printf("   w    \n");
        printf(" a s d  \n");
        printf("\n");
        printf("w/s : increase/decrease linear velocity\n");
        printf("a/d : increase/decrease angular velocity (turn left/right)\n");
        printf("space : stop immediately\n");
        printf("q : quit\n");
        printf("\n");
        printf("Current velocities:\n");
        printf("Linear: %.2f m/s (max: %.1f)\n", linear_vel_, linear_speed_);
        printf("Angular: %.2f rad/s (max: %.1f)\n", angular_vel_, angular_speed_);
        printf("\n");
    }
    
    void processKey(char key)
    {
        bool velocity_changed = false;
        
        switch (key) {
            case 'w':
                linear_vel_ = std::min(linear_vel_ + linear_step_, linear_speed_);
                velocity_changed = true;
                break;
            case 's':
                linear_vel_ = std::max(linear_vel_ - linear_step_, -linear_speed_);
                velocity_changed = true;
                break;
            case 'a':
                angular_vel_ = std::min(angular_vel_ + angular_step_, angular_speed_);
                velocity_changed = true;
                break;
            case 'd':
                angular_vel_ = std::max(angular_vel_ - angular_step_, -angular_speed_);
                velocity_changed = true;
                break;
            case ' ':
                linear_vel_ = 0.0;
                angular_vel_ = 0.0;
                velocity_changed = true;
                RCLCPP_INFO(this->get_logger(), "Emergency stop!");
                break;
            case 'q':
                RCLCPP_INFO(this->get_logger(), "Shutting down teleop controller...");
                rclcpp::shutdown();
                return;
            default:
                break;
        }
        
        if (velocity_changed) {
            printf("\rLinear: %6.2f m/s | Angular: %6.2f rad/s", linear_vel_, angular_vel_);
            fflush(stdout);
        }
    }
    
    void publishVelocity()
    {
        auto twist_msg = geometry_msgs::msg::Twist();
        twist_msg.linear.x = linear_vel_;
        twist_msg.angular.z = angular_vel_;
        cmd_vel_pub_->publish(twist_msg);
    }
};

static void signalHandler(int sig)
{
    (void)sig;
    rclcpp::shutdown();
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    // Set up signal handler for clean shutdown
    signal(SIGINT, signalHandler);
    
    auto node = std::make_shared<USVTeleopKeyboard>();
    
    try {
        node->run();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("usv_teleop_keyboard"), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}
