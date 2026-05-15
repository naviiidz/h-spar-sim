from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.actions import OpaqueFunction

def generate_multi_robot_nodes(context, *args, **kwargs):
    """Generate nodes for multiple robots based on robot_names parameter"""
    robot_names = LaunchConfiguration('robot_names').perform(context).split(',')
    world_name = LaunchConfiguration('world_name').perform(context)
    max_thrust = LaunchConfiguration('max_thrust').perform(context)
    
    nodes = []
    
    for robot_name in robot_names:
        robot_name = robot_name.strip()  # Remove any whitespace
        
        # ROS-Gazebo Bridge for left thruster
        nodes.append(Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name=f'{robot_name}_left_thruster_bridge',
            arguments=[
                f'/{robot_name}/thrusters/left/thrust@std_msgs/msg/Float64@gz.msgs.Double'
            ],
            remappings=[
                (f'/{robot_name}/thrusters/left/thrust', f'/{robot_name}/thrusters/left/thrust')
            ],
            output='screen'
        ))
        
        # ROS-Gazebo Bridge for right thruster
        nodes.append(Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name=f'{robot_name}_right_thruster_bridge',
            arguments=[
                f'/{robot_name}/thrusters/right/thrust@std_msgs/msg/Float64@gz.msgs.Double'
            ],
            remappings=[
                (f'/{robot_name}/thrusters/right/thrust', f'/{robot_name}/thrusters/right/thrust')
            ],
            output='screen'
        ))
        
        # ROS-Gazebo Bridge for cmd_vel
        nodes.append(Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name=f'{robot_name}_cmd_vel_bridge',
            arguments=[
                f'/{robot_name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
            ],
            output='screen'
        ))
        
        # ROS-Gazebo Bridge for cameras
        nodes.append(Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name=f'{robot_name}_camera_bridge',
            arguments=[
                f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/front_left_camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image',
                f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/front_left_camera_sensor/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
                f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/front_right_camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image',
                f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/front_right_camera_sensor/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
                f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/middle_right_camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image',
                f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/middle_right_camera_sensor/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'
            ],
            remappings=[
                (f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/front_left_camera_sensor/image', f'/{robot_name}/cameras/front_left/image'),
                (f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/front_left_camera_sensor/camera_info', f'/{robot_name}/cameras/front_left/camera_info'),
                (f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/front_right_camera_sensor/image', f'/{robot_name}/cameras/front_right/image'),
                (f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/front_right_camera_sensor/camera_info', f'/{robot_name}/cameras/front_right/camera_info'),
                (f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/middle_right_camera_sensor/image', f'/{robot_name}/cameras/middle_right/image'),
                (f'/world/{world_name}/model/{robot_name}/link/{robot_name}/base_link/sensor/middle_right_camera_sensor/camera_info', f'/{robot_name}/cameras/middle_right/camera_info'),
            ],
            output='screen'
        ))
        
        # Pose and TF Publisher for each robot
        nodes.append(Node(
            package='usv_control',
            executable='gz_dynamic_pose_listener.py',
            name=f'{robot_name}_gz_dynamic_pose_listener',
            namespace=robot_name,
            output='screen',
            parameters=[{
                'world_name': world_name,
                'robot_name': robot_name
            }]
        ))
        
        # Velocity controller for each robot
        nodes.append(Node(
            package='usv_control',
            executable='usv_velocity_controller',
            name='usv_velocity_controller',
            namespace=robot_name,
            output='screen',
            parameters=[{
                'max_thrust': float(max_thrust),
                'boat_length': 4.9,
                'thrust_deadband': 0.1,
            }],
            remappings=[
                ('/wamv/cmd_vel', f'/{robot_name}/cmd_vel'),
                ('/wamv/thrusters/left/thrust', f'/{robot_name}/thrusters/left/thrust'),
                ('/wamv/thrusters/right/thrust', f'/{robot_name}/thrusters/right/thrust')
            ]
        ))
    
    return nodes

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'robot_names',
            default_value='wamv1,wamv2,wamv3,wamv4',
            description='Comma-separated list of robot names'
        ),
        
        DeclareLaunchArgument(
            'world_name',
            default_value='baylands',
            description='Name of the Gazebo world'
        ),
        
        DeclareLaunchArgument(
            'max_thrust',
            default_value='250.0',
            description='Maximum thrust for each thruster (Newtons)'
        ),
        
        # Generate nodes for all robots
        OpaqueFunction(function=generate_multi_robot_nodes),
    ])
