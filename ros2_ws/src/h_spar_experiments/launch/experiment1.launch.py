from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Locate the usv_control launch file
    usv_control_share = get_package_share_directory('usv_control')
    usv_bringup = os.path.join(usv_control_share, 'launch', 'usv_bringup.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument('field', default_value='sydney_regatta_roi_from1326s',
                              description='velocity_fields/<field> used for the current'),
        DeclareLaunchArgument('field_time_offset_s', default_value='1200.0',
                              description='field time when the Gazebo clock reads 0'),
        DeclareLaunchArgument('current_enabled', default_value='true',
                              description='false: no current (VRX baseline)'),
        # Bridge the water current to VRX SimpleHydrodynamics in Gazebo
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='water_current_bridge',
            arguments=['/wamv/water_current@geometry_msgs/msg/Vector3]gz.msgs.Vector3d'],
            output='screen'
        ),

        # Drag query service
        Node(
            package='drag_query',
            executable='drag_query_server.py',
            name='drag_query_server',
            output='screen',
            parameters=[{
                'field': LaunchConfiguration('field'),
                'field_time_offset_s': LaunchConfiguration('field_time_offset_s'),
                'current_enabled': LaunchConfiguration('current_enabled'),
            }],
        ),

        # Include the usv_control bringup launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(usv_bringup)
        ),


        # Experimental node that queries and publishes the water current
        Node(
            package='h_spar_experiments',
            executable='drag_force_node',
            name='drag_force_node',
            output='screen'
        ),
    ])
