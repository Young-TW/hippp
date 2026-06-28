cmake_minimum_required(VERSION 3.21)

project(hippp LANGUAGES CXX VERSION 0.0.0)

include_directories(include)
set(CMAKE_CXX_STANDARD 23)

# Probe for a HIP-capable compiler without hard-failing when ROCm is absent.
# check_language() runs cmake in a subprocess; if that subprocess exits non-zero
# (e.g. because CMakeDetermineHIPCompiler.cmake cannot find the ROCm root) it
# silently reports CMAKE_HIP_COMPILER-NOTFOUND instead of issuing a FATAL_ERROR.
include(CheckLanguage)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    add_definitions(-D__HIP_PLATFORM_AMD__)
    set(HIP_AVAILABLE TRUE)
else()
    message(STATUS
        "HIP compiler not found — GPU executable skipped; "
        "regression test compiled against CPU mock")
    set(HIP_AVAILABLE FALSE)
endif()

if(HIP_AVAILABLE)
    add_executable(hippp
        src/main.cpp
        src/example/origin/vector_add.hip
        src/example/origin/matrix_multiplication.hip
        src/example/raii/vector_add.hip
        src/example/raii/matrix_multiplication.hip
        src/example/device/vector_add.hip
        src/example/device/matrix_multiplication.hip
    )
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

# Regression test for issue #12: tiled shared-memory SGEMM correctness
enable_testing()
add_executable(test_matrix_multiplication_tiled
    test/test_matrix_multiplication_tiled.hip
    src/example/device/matrix_multiplication.hip
)
if(HIP_AVAILABLE)
    set_source_files_properties(
        test/test_matrix_multiplication_tiled.hip
        src/example/device/matrix_multiplication.hip
        PROPERTIES LANGUAGE HIP
    )
    target_include_directories(test_matrix_multiplication_tiled PRIVATE include)
else()
    # Compile HIP sources as C++ using a CPU mock of the HIP runtime.
    # The mock runs each GPU thread in a real std::thread so __shared__ /
    # __syncthreads() semantics are preserved and the test is not weakened.
    set_source_files_properties(
        test/test_matrix_multiplication_tiled.hip
        src/example/device/matrix_multiplication.hip
        PROPERTIES LANGUAGE CXX
    )
    target_include_directories(test_matrix_multiplication_tiled PRIVATE
        include
        test/mock_hip   # provides hip/hip_runtime.h via CPU simulation
    )
endif()
add_test(NAME matrix_multiplication_tiled
         COMMAND test_matrix_multiplication_tiled)
