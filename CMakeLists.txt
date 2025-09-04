cmake_minimum_required(VERSION 3.21)

project(hippp LANGUAGES CXX HIP VERSION 0.0.0)
add_definitions(-D__HIP_PLATFORM_AMD__)

include_directories(include)

set(CMAKE_CXX_STANDARD 23)

# Specify the HIP compiler path
set(HIP_COMPILER /opt/rocm/bin/hipcc)
set(CMAKE_HIP_COMPILER /opt/rocm/bin/hipcc)

set(HIP_ENABLE_LANG OFF)
# HIP toolchain settings
find_package(HIP REQUIRED)

# Add HIP source files
add_executable(hippp
    src/main.cpp
    src/example/origin/vector_add.hip
    src/example/origin/matrix_multiplication.hip
    src/example/raii/vector_add.hip
    src/example/raii/matrix_multiplication.hip
    src/example/device/vector_add.hip
    src/example/device/matrix_multiplication.hip
)

# Set HIP as the language for .hip files
set_source_files_properties(
    src/main.cpp
    src/example/origin/vector_add.hip
    src/example/origin/matrix_multiplication.hip
    src/example/raii/vector_add.hip
    src/example/raii/matrix_multiplication.hip
    src/example/device/vector_add.hip
    src/example/device/matrix_multiplication.hip
    PROPERTIES LANGUAGE HIP
)

# Add Catch2 for testing
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.4
)
FetchContent_MakeAvailable(Catch2)

enable_testing()

add_executable(hippp_tests
    tests/test_main.cpp
)

target_include_directories(hippp_tests PRIVATE /opt/rocm/include)

target_link_libraries(hippp_tests PRIVATE Catch2::Catch2WithMain)

add_test(NAME hippp_tests COMMAND hippp_tests)
