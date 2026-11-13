from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Locate the usv_control launch file
    usv_control_share = get_package_share_directory('usv_control')
    usv_bringup = os.path.join(usv_control_share, 'launch', 'usv_bringup.launch.py')

    return LaunchDescription([
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
            output='screen'
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
