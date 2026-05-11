cmake_minimum_required(VERSION 3.8)
project(h_spar_force)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)

add_executable(ros_to_gazebo_wrench_bridge src/apply_link_force_node.cpp)

ament_target_dependencies(ros_to_gazebo_wrench_bridge
  rclcpp
  geometry_msgs
)

target_link_libraries(ros_to_gazebo_wrench_bridge
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10
)

install(TARGETS
  ros_to_gazebo_wrench_bridge
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
