cmake_minimum_required(VERSION 2.8.3)
project(visualization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  hand_detection
  finger_analysis
  camera_calibration
  pointing_localization
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs hand_detection finger_analysis camera_calibration pointing_localization
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
