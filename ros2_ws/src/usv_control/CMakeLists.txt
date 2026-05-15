cmake_minimum_required(VERSION 3.8)
project(usv_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# Add executable for teleop keyboard controller
add_executable(usv_teleop_keyboard src/usv_teleop_keyboard.cpp)
ament_target_dependencies(usv_teleop_keyboard 
  rclcpp 
  geometry_msgs
)

# Add executable for velocity controller
add_executable(usv_velocity_controller src/usv_velocity_controller.cpp)
ament_target_dependencies(usv_velocity_controller 
  rclcpp 
  geometry_msgs
  std_msgs
)

# Add executable for path tracker
add_executable(usv_path_tracker src/usv_path_tracker.cpp)
ament_target_dependencies(usv_path_tracker
  rclcpp
  geometry_msgs
  nav_msgs
)

# Install executables
install(TARGETS
  usv_teleop_keyboard
  usv_velocity_controller
  usv_path_tracker
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python scripts
install(PROGRAMS
  scripts/gz_dynamic_pose_listener.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
