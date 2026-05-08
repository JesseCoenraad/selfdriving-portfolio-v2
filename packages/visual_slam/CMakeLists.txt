cmake_minimum_required(VERSION 2.8.3)
project(visual_slam)

find_package(catkin REQUIRED COMPONENTS
  duckietown_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  rospy
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
