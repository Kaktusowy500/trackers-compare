add_library(trackers CSRTTracker.cpp)

target_include_directories(trackers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(trackers ${OpenCV_LIBS})
