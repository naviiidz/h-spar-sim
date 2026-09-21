#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    robot_frame_arg = DeclareLaunchArgument(
        'robot_frame',
        default_value='wamv_simple',
        description='Robot base frame'
    )
    
    global_frame_arg = DeclareLaunchArgument(
        'global_frame',
        default_value='odom',
        description='Global frame for navigation'
    )
    
    # Simple DWA Planner Node
    dwa_node = Node(
        package='usv_planners',
        executable='simple_dwa_planner',
        name='simple_dwa_planner',
        parameters=[{
            'robot_frame': 'wamv_simple',
            'global_frame': 'map',
            'goal_tolerance': 2.0,     # Increased tolerance for faster goal reaching
            'max_vel_x': 50.0,         # Keep high max velocity
            'min_vel_x': 2.0,          # Increased minimum velocity (no slow crawling)
            'max_vel_theta': 5.0,      # Much faster rotation
            'min_vel_theta': -5.0,     # Much faster rotation
            'acc_lim_x': 10.0,         # Much faster acceleration (was 2.0)
            'acc_lim_theta': 8.0,      # Much faster angular acceleration (was 3.0)
            'sim_time': 1.0,           # Reduced prediction time for faster response (was 2.0)
            'sim_granularity': 0.2,    # Increased granularity for faster computation (was 0.1)
            'vx_samples': 5,           # Fewer velocity samples for speed (was 10)
            'vyaw_samples': 10         # Fewer angular samples for speed (was 20)
        }],
        remappings=[
            ('odom', '/odom'),
            ('cmd_vel', '/cmd_vel'),
            ('goal_pose', '/goal_pose')
        ],
        output='screen'
    )
    
    # Path Visualizer Node
    path_visualizer_node = Node(
        package='usv_planners',
        executable='path_visualizer.py',
        name='path_visualizer',
        parameters=[{
            'robot_frame': 'wamv_simple',
            'global_frame': 'map',
            'path_color_r': 0.0,
            'path_color_g': 1.0,
            'path_color_b': 0.0,
            'path_color_a': 1.0,
            'path_width': 0.5,
            'max_path_points': 1000,
            'min_distance_threshold': 0.5,
            'publish_rate': 10.0
        }],
        output='screen'
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        robot_frame_arg,
        global_frame_arg,
        dwa_node,
        path_visualizer_node
    ])
