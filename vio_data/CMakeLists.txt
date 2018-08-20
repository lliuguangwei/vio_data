cmake_minimum_required(VERSION 2.8.3)
project(empty_ros_pkg)

set(CMAKE_CXX_STANDARD 11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
   cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  novatel_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES empty_ros_pkg
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

# find_package(Eigen3 3.1.0 REQUIRED)
find_package(Eigen3 REQUIRED)

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
)

# Node for monocular camera
add_executable(Mono ros_mono.cc)
add_dependencies(Mono ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(Mono ${catkin_LIBRARIES})

# target_link_libraries(Mono
# ${LIBS}
# )


