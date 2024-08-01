add_library(trackers
ITracker.cpp
CSRTTracker.cpp
VITTracker.cpp
DaSiamTracker.cpp)

target_include_directories(trackers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(trackers ${OpenCV_LIBS})
