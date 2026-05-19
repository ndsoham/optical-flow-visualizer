cmake_minimum_required(VERSION 3.15)
project(optical_flow_visualizer)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(optical_flow_visualizer main.cpp)

target_link_libraries(optical_flow_visualizer PRIVATE ${OpenCV_LIBS})
target_include_directories(optical_flow_visualizer PRIVATE ${OpenCV_INCLUDE_DIRS})