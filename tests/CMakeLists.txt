include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_dataset_utils test_dataset_utils.cpp)

target_link_libraries(test_dataset_utils gtest_main utils)

include(GoogleTest)
gtest_discover_tests(test_dataset_utils)
