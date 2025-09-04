cmake_minimum_required(VERSION 3.21)

# Check if we're building in testing mode (without HIP hardware)
option(BUILD_TESTS_ONLY "Build only tests with mock HIP" OFF)

if(BUILD_TESTS_ONLY)
    project(hippp LANGUAGES CXX VERSION 0.0.0)
    set(CMAKE_CXX_STANDARD 23)
    include_directories(include)
    include_directories(tests/mock_hip)
else()
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
endif()

# Add HIP source files (only when not in test-only mode)
if(NOT BUILD_TESTS_ONLY)
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
endif()

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

if(BUILD_TESTS_ONLY)
    target_include_directories(hippp_tests PRIVATE tests/mock_hip)
else()
    target_include_directories(hippp_tests PRIVATE /opt/rocm/include)
endif()

target_link_libraries(hippp_tests PRIVATE Catch2::Catch2WithMain)

add_test(NAME hippp_tests COMMAND hippp_tests)
