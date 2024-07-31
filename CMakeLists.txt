cmake_minimum_required(VERSION 3.10)

# Define the project
project(trackers-compare)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add executable using the project name
add_executable(${PROJECT_NAME} main.cpp)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
