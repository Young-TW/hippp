# hippp - HIP programming with RAII

hippp is a C++ wrapper for the HIP API, designed to simplify the process of writing GPU code. It provides a RAII (Resource Acquisition Is Initialization) style interface, making it easier to manage resources and ensuring that they are properly released when no longer needed. This library is particularly useful for developers who want to write portable GPU code without dealing with the complexities of the HIP API directly.

![CI/CD](https://github.com/Young-TW/hippp/actions/workflows/cmake.yml/badge.svg)

## Features

- [x] **Device Memory** (`hipMalloc` / `hipFree`) → `HipBuffer<T>`
- [x] **Pinned Host Memory** (`hipHostMalloc` / `hipHostFree`) → `HipHostBuffer<T>`
- [x] **Streams** (`hipStreamCreate` / `hipStreamDestroy`) → `HipStream`
- [x] **Events** (`hipEventCreate` / `hipEventDestroy`) → `HipEvent`
- [ ] **Graphs** (`hipGraphCreate` / `hipGraphDestroy`) → `HipGraph`
- [ ] **Graph Executable** (`hipGraphExecCreate` / `hipGraphExecDestroy`) → `HipGraphExec`
- [ ] **Modules** (`hipModuleLoad` / `hipModuleUnload`, `hipModuleGetFunction`) → `HipModule`, `HipFunction`
- [x] **Kernels** (`hipLaunchKernel`) → `HipKernel`
- [ ] **Contexts** (`hipCtxCreate` / `hipCtxDestroy`) → `HipContext`
- [ ] **Arrays** (`hipArrayCreate` / `hipArrayDestroy`, `hipMipmappedArrayCreate` / `hipMipmappedArrayDestroy`) → `HipArray`, `HipMipmappedArray`
- [ ] **Texture Objects** (`hipTexObjectCreate` / `hipTexObjectDestroy`) → `HipTextureObject`
- [ ] **Surface Objects** (`hipSurfObjectCreate` / `hipSurfObjectDestroy`) → `HipSurfaceObject`
- [ ] **Memory Pools** (`hipMemPoolCreate` / `hipMemPoolDestroy`) → `HipMemPool`
- [ ] **External Library Handles** (rocBLAS, rocFFT, rocRAND) → `RocblasHandle`, `RocfftPlan`, `RocrandGenerator`

## Requirements

- Compiler of C++17 or later
- hipcc (AMD ROCm)
- CMake 3.10 or later
- A compatible GPU (AMD Radeon or AMD Instinct, or NVIDIA with HIP support)

## Benchmark

| Tasks | Size | GPU | Origin | RAII |
|-------|------|-----|--------|------|
| Vector Addition | 1,048576 | RX 7600 XT | 2003 µs | 1995 µs |
| Matrix Multiplication | 1,024 x 1,024 | RX 7600 XT | 7215 µs | 8008 µs |
