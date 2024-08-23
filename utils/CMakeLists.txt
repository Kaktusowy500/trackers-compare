add_library(utils
    DatasetUtils.cpp)
target_include_directories(utils PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(utils PUBLIC ${OpenCV_LIBS})
