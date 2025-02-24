cmake_minimum_required(VERSION 3.0.2)
project(rosneuro_dl)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  rosneuro_msgs
)


find_package(PkgConfig)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES filters_tutorial
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  include
)

## Declare a python executable


# catkin_install_python(PROGRAMS scripts/predict_neural.py scripts/filter_chain.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

