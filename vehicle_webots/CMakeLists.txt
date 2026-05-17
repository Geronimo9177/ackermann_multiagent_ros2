cmake_minimum_required(VERSION 3.8)
project(vehicle_webots)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(webots_ros2_driver REQUIRED)
find_package(pluginlib REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)


# ── Driver plugin ─────────────────────────────────────────────────
add_library(vehicle_webots_driver SHARED
  src/CarDriver.cpp
)

target_include_directories(vehicle_webots_driver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(vehicle_webots_driver
  rclcpp
  webots_ros2_driver
  pluginlib
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2_ros
  tf2
)

pluginlib_export_plugin_description_file(
  webots_ros2_driver vehicle_webots_plugin.xml)

install(TARGETS vehicle_webots_driver
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

# ── Install launch files, worlds, and models ─────────────────────────────────────────────────
install(
  DIRECTORY config launch worlds models
  DESTINATION share/${PROJECT_NAME}
)

install(FILES vehicle_webots_plugin.xml
  DESTINATION share/${PROJECT_NAME}
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
