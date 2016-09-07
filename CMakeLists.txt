cmake_minimum_required(VERSION 3.5)
project(pano)

find_package(OpenCV REQUIRED)
include_directories(include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES src/main.cpp src/FeaturesFinder.cpp include/FeaturesFinder.hpp src/SiftFeaturesFinder.cpp include/SiftFeaturesFinder.hpp src/Util.cpp include/Util.hpp)
add_executable(pano ${SOURCE_FILES})
target_link_libraries( pano ${OpenCV_LIBS} )
