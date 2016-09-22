cmake_minimum_required(VERSION 3.5)
project(pano)

find_package(OpenCV REQUIRED)
include_directories(include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES src/main.cpp src/FeatureFinder.cpp include/FeatureFinder.hpp src/SiftFeatureFinder.cpp include/SiftFeatureFinder.hpp src/Util.cpp include/Util.hpp src/FeatureMatcher.cpp include/FeatureMatcher.hpp src/OrbFeatureFinder.cpp include/OrbFeatureFinder.hpp src/SurfFeatureFinder.cpp include/SurfFeatureFinder.hpp)
add_executable(pano ${SOURCE_FILES})
target_link_libraries( pano ${OpenCV_LIBS} )
