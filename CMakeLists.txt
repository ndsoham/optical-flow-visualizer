cmake_minimum_required(VERSION 3.15)
project(optical_flow_visualizer)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(OpenMP)

add_executable(optical_flow_visualizer main.cpp)

target_link_libraries(optical_flow_visualizer PRIVATE ${OpenCV_LIBS})
target_include_directories(optical_flow_visualizer PRIVATE ${OpenCV_INCLUDE_DIRS})


if (OpenMP_CXX_FOUND)
    target_link_libraries(optical_flow_visualizer PRIVATE OpenMP::OpenMP_CXX)
endif()

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    include_directories("/opt/homebrew/Cellar/libomp/22.1.6/include/")
    target_link_libraries(optical_flow_visualizer PRIVATE "/opt/homebrew/Cellar/libomp/22.1.6/lib/libomp.dylib")
endif()


