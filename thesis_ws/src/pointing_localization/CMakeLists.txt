cmake_minimum_required(VERSION 2.8.3)
project(pointing_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  finger_analysis
  camera_calibration
  message_generation
)

add_message_files(
  FILES
  PointingTarget.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  finger_analysis
  camera_calibration
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs finger_analysis camera_calibration message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
