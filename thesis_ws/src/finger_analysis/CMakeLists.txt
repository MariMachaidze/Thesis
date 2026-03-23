cmake_minimum_required(VERSION 2.8.3)
project(finger_analysis)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  hand_detection
  message_generation
)

add_message_files(
  FILES
  FingerAnalysis.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  hand_detection
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs hand_detection message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
