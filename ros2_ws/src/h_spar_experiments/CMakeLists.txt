cmake_minimum_required(VERSION 3.8)
project(h_spar_experiments)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install resource marker for this package
install(DIRECTORY
  resource/
  DESTINATION share/ament_index/resource_index/packages
  FILES_MATCHING PATTERN "${PROJECT_NAME}")

# Install entry points
install(PROGRAMS
  h_spar_experiments/drag_force_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME drag_force_node
)

ament_package()
