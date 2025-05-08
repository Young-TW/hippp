cmake_minimum_required(VERSION 3.21)

project(hippp LANGUAGES CXX HIP VERSION 0.0.0)
add_definitions(-D__HIP_PLATFORM_AMD__)

include_directories(include)

# 指定 HIP 編譯器路徑
set(HIP_COMPILER clang)
set(CMAKE_HIP_COMPILER /opt/rocm/bin/hipcc)

# HIP 工具鏈設定
find_package(HIP REQUIRED)

# 加入 HIP 原始碼
add_executable(hippp
    src/main.cpp
    src/example/vector_add.hip
)

# 標記為 HIP 檔案
set_source_files_properties(
    src/main.cpp
    src/example/vector_add.hip
    PROPERTIES LANGUAGE HIP
)
