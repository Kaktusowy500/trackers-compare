add_library(evaluation
TrackerPerformanceEvaluator.cpp
)

target_include_directories(evaluation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(evaluation ${OpenCV_LIBS} spdlog::spdlog)
