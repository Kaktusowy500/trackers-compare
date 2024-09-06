cmake_minimum_required(VERSION 3.10)

project(tracker_compare)
set(OpenCV_DIR ~/git/build) # use opencv 4.9 built from source
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(trackers)
add_subdirectory(utils)
add_subdirectory(tests)
add_subdirectory(evaluation)
add_subdirectory(spdlog)

add_executable(${PROJECT_NAME} tracker_compare.cpp TrackerComparator.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} utils trackers evaluation spdlog::spdlog yaml-cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} utils trackers evaluation)
