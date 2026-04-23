cmake_minimum_required(VERSION 2.8.3)
project(sphero_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  camera_calibration
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs camera_calibration
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
