cmake_minimum_required(VERSION 3.8)
project(usv_planners)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

# Include directories
include_directories(include)

# Simple DWA Planner Node
add_executable(simple_dwa_planner src/simple_dwa_planner.cpp)
ament_target_dependencies(simple_dwa_planner
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
  tf2_ros
  visualization_msgs
)

# Install executables
install(TARGETS
  simple_dwa_planner
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts as executables
install(PROGRAMS
  scripts/send_goal.py
  scripts/path_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/sequential_goal_sender.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME sequential_goal_sender
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch/
)

# Install scripts directory
install(DIRECTORY scripts/
  DESTINATION share/${PROJECT_NAME}/scripts/
  USE_SOURCE_PERMISSIONS
)

# Install header files
install(DIRECTORY include/
  DESTINATION include/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
