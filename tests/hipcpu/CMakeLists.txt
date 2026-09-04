# Standalone unit tests for hippp running on AMD's official HIP CPU Runtime
# (https://github.com/ROCm/HIP-CPU) — no GPU and no ROCm installation needed.
# HIP-CPU is fetched automatically from the ROCm GitHub org.
#
# Requirements: C++17-or-later host compiler, Intel oneTBB (e.g. `apt install
# libtbb-dev`), CMake >= 3.14, git, and network access at configure time.
#
# Usage (from the repository root):
#   cmake -S tests/hipcpu -B build-hipcpu -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-hipcpu -j
#   ctest --test-dir build-hipcpu --output-on-failure
cmake_minimum_required(VERSION 3.14)

project(hippp_hipcpu_tests LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
    hip_cpu_rt
    GIT_REPOSITORY https://github.com/ROCm/HIP-CPU.git
    GIT_TAG        master  # Repository archived 2026-07; master is final.
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(hip_cpu_rt)

add_executable(hip_raii_test
    hip_raii_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/example/device/vector_add.hip
)

# The host compiler does not know the .hip extension; force C++.
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/example/device/vector_add.hip
    PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-x;c++"
)

set_target_properties(hip_raii_test PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_include_directories(
    hip_raii_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(hip_raii_test PRIVATE hip_cpu_rt::hip_cpu_rt)

enable_testing()
add_test(NAME hip_raii_test COMMAND hip_raii_test)
