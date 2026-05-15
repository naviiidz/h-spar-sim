from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'max_thrust',
            default_value='250.0',
            description='Maximum thrust for each thruster (Newtons)'
        ),
        
        DeclareLaunchArgument(
            'world_name',
            default_value='sydney_regatta',
            description='Name of the Gazebo world'
        ),
        
        # ROS-Gazebo Bridge for left thruster
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='left_thruster_bridge',
            arguments=[
                '/wamv/thrusters/left/thrust@std_msgs/msg/Float64@gz.msgs.Double'
            ],
            output='screen'
        ),
        
        # ROS-Gazebo Bridge for right thruster
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='right_thruster_bridge',
            arguments=[
                '/wamv/thrusters/right/thrust@std_msgs/msg/Float64@gz.msgs.Double'
            ],
            output='screen'
        ),
        
        # ROS-Gazebo Bridge for cmd_vel (optional)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='cmd_vel_bridge',
            arguments=[
                '/wamv/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
            ],
            output='screen'
        ),
        
        # WAMV Pose and TF Publisher (extracts WAMV pose from Gazebo and publishes TF)
        Node(
            package='usv_control',
            executable='gz_dynamic_pose_listener.py',
            name='gz_dynamic_pose_listener',
            output='screen',
            parameters=[{
                'world_name': LaunchConfiguration('world_name'),
                'robot_name': 'wamv'
            }]
        ),
        
        # Velocity controller node
        Node(
            package='usv_control',
            executable='usv_velocity_controller',
            name='usv_velocity_controller',
            output='screen',
            parameters=[{
                'max_thrust': LaunchConfiguration('max_thrust'),
                'boat_length': 4.9,
                'thrust_deadband': 0.1,
            }]
        )
    ])
