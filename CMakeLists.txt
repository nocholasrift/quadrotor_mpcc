cmake_minimum_required(VERSION 3.5.0)
project(mpcc C CXX)

# add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AcadosUtils)


## Try to find catkin (optional)
find_package(catkin QUIET COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  trajectory_msgs
  grid_map_ros
  costmap_2d
  amrl_logging
)

# find_package(pybind11 QUIET)
find_package(Python3 REQUIRED)
find_package(ACADOS REQUIRED)
find_package(BLASFEO REQUIRED)
find_package(HPIPM REQUIRED)
find_package(Eigen3 REQUIRED)


add_subdirectory(third_party/pybind11)

message(STATUS "ACADOS_INCLUDE_DIRS: ${ACADOS_INCLUDE_DIRS}")

########################
# Catkin-specific stuff
########################
if(catkin_FOUND)
  add_definitions(-DFOUND_CATKIN)
  message(STATUS "Catkin found, building ROS nodes")

  add_message_files(
    DIRECTORY
    FILES
      RLState.msg
  )

  add_service_files(
    FILES
      QuerySAC.srv
      QuerySACDI.srv
  )

  generate_messages(
    DEPENDENCIES 
      nav_msgs
      geometry_msgs 
      trajectory_msgs
  )

  catkin_package(
    CATKIN_DEPENDS message_runtime
  )
else()
  message(STATUS "Catkin not found, skipping ROS message/service generation")
endif()

if (APPLE)
  set(EIGEN3_INCLUDE_DIRS /opt/homebrew/opt/eigen/include/eigen3)
endif()

########################
# Common include paths
########################
include_directories(
  include
  ${HPIPM_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${ACADOS_INCLUDE_DIRS}
  ${BLASFEO_INCLUDE_DIRS}
)

if(catkin_FOUND)
  include_directories(${catkin_INCLUDE_DIRS})
endif()

########################
# MPCC Core Library
########################
add_library(mpcc_core
  src/mpcc_core.cpp
)
target_compile_options(mpcc_core PRIVATE -std=c++17 -O2)
target_include_directories(mpcc_core PRIVATE 
  # ${CASADI_GEN_DIR}
  # ${ACADOS_OUTPUT_DIRS} 
  # ${HiGHS_INCLUDE_DIRS}
)

target_link_libraries(mpcc_core
  ${HPIPM_LIBRARIES}
  ${ACADOS_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${ACADOS_OUTPUT_LIBS}
  ${CASADI_OUTPUT_LIBS}
  ${BLASFEO_LIBRARIES}
  # acados_generated_lib
  # highs
)

########################
# Python bindings
########################
# message(STATUS "Pybind11 v${pybind11_VERSION} found, building mpcc python bindings")
# add_definitions(-DFOUND_PYBIND11)
#
# find_package(Python3 COMPONENTS Development REQUIRED)
#
# pybind11_add_module(py_mpcc
#   src/mpcc/py_mpcc.cpp
# )
#
# target_include_directories(py_mpcc PRIVATE 
#   ${ACADOS_OUTPUT_DIRS} 
#   ${PYTHON_INCLUDE_DIRS}
# )
#
# target_compile_options(py_mpcc PRIVATE -std=c++17 -O2)
# target_link_libraries(py_mpcc
#   PRIVATE
#   mpcc_core
#   ${CASADI_OUTPUT_LIBS}
# )

########################
# ROS executables (only if catkin)
########################
if(catkin_FOUND)

  add_executable(mpcc_ros 
    src/mpcc/ros/logger.cpp
    src/mpcc/ros/mpcc_ros.cpp
    src/mpcc/ros/mpcc_node.cpp
  )
  target_link_libraries(mpcc_ros
    ${catkin_LIBRARIES}
    mpcc_core
  )
  # add_dependencies(mpcc_ros ${PROJECT_NAME}_generate_messages ${ACADOS_TARGETS})
  target_compile_options(mpcc_ros PRIVATE -std=c++17 -O2)
  target_include_directories(mpcc_ros PRIVATE ${ACADOS_OUTPUT_DIRS})

  add_executable(publish_pf_pose src/utils/publish_pf_pose.cpp)
  target_link_libraries(publish_pf_pose ${catkin_LIBRARIES})

  add_executable(local_costmap src/utils/local_costmap.cpp)
  target_link_libraries(local_costmap ${catkin_LIBRARIES})
endif()

