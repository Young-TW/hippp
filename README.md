# hippp - HIP programming with RAII

hippp is a C++ wrapper for the HIP API, designed to simplify the process of writing GPU code. It provides a RAII (Resource Acquisition Is Initialization) style interface, making it easier to manage resources and ensuring that they are properly released when no longer needed. This library is particularly useful for developers who want to write portable GPU code without dealing with the complexities of the HIP API directly.

![CI/CD](https://github.com/Young-TW/hippp/actions/workflows/cmake.yml/badge.svg)

## Features

- [X] **Device Memory** (`hipMalloc` / `hipFree`) → `HipBuffer<T>`
- [X] **Pinned Host Memory** (`hipHostMalloc` / `hipHostFree`) → `HipHostBuffer<T>`
- [X] **Streams** (`hipStreamCreate` / `hipStreamDestroy`) → `HipStream`
- [X] **Events** (`hipEventCreate` / `hipEventDestroy`) → `HipEvent`
- [ ] **Graphs** (`hipGraphCreate` / `hipGraphDestroy`) → `HipGraph`
- [ ] **Graph Executable** (`hipGraphExecCreate` / `hipGraphExecDestroy`) → `HipGraphExec`
- [X] **Modules** (`hipModuleLoad` / `hipModuleUnload`, `hipModuleGetFunction`) → `HipModule`, `HipFunction`
- [X] **Kernels** (`hipLaunchKernel`) → `HipKernel`
- [X] **Contexts** (`hipCtxCreate` / `hipCtxDestroy`) → `HipContext`
- [ ] **Arrays** (`hipArrayCreate` / `hipArrayDestroy`, `hipMipmappedArrayCreate` / `hipMipmappedArrayDestroy`) → `HipArray`, `HipMipmappedArray`
- [ ] **Texture Objects** (`hipTexObjectCreate` / `hipTexObjectDestroy`) → `HipTextureObject`
- [ ] **Surface Objects** (`hipSurfObjectCreate` / `hipSurfObjectDestroy`) → `HipSurfaceObject`
- [X] **Memory Pools** (`hipMallocAsync` / `hipFreeAsync`, default pool) → `HipMemPool` (explicit `hipMemPoolCreate` custom pools not included)
- [ ] **External Library Handles** (rocBLAS, rocFFT, rocRAND) → `RocblasHandle`, `RocfftPlan`, `RocrandGenerator`

## Requirements

- Compiler of C++17 or later
- hipcc (AMD ROCm)
- CMake 3.10 or later
- A compatible GPU (AMD Radeon or AMD Instinct, or NVIDIA with HIP support)

## Unit Tests (no GPU required)

Unit tests for the RAII wrappers (`HipBuffer`, `HipHostBuffer`, `HipStream`, `HipEvent`, `HipKernel`, `HipContext`, `HipModule`/`HipFunction`, `HipMemPool`) run on AMD's official [HIP CPU Runtime](https://github.com/ROCm/HIP-CPU), which executes unmodified HIP code on the CPU. Neither a GPU nor a ROCm installation is needed — only a C++ compiler, Intel oneTBB (`libtbb-dev`), and CMake ≥ 3.14:

```bash
cmake -S tests/hipcpu -B build-hipcpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-hipcpu -j
ctest --test-dir build-hipcpu --output-on-failure
```

Note: HIP-CPU is archived and implements only a subset of the HIP API; wrappers that need unsupported APIs (arrays, textures, surfaces, graphs, explicit memory pools, external libraries) are not covered by these tests. Module-loaded kernels additionally cannot take arguments and run one fiber per block under HIP-CPU (data exchange through module globals); on GPU HIP the same `HipFunction` API passes arguments normally.

## Benchmark

| Tasks                 | Size          | GPU        | ROCm Version | Origin   | RAII     |
| --------------------- | ------------- | ---------- | ------------ | -------- | -------- |
| Vector Addition       | 1,048576      | RX 7600 XT | 6.3.0        | 2003 µs | 1995 µs |
| Matrix Multiplication | 1,024 x 1,024 | RX 7600 XT | 6.3.0        | 7215 µs | 8008 µs |
| Vector Addition       | 1,048576      | RX 9070 XT | 7.0.0        | 1214 µs | 1187 µs |
| Matrix Multiplication | 1,024 x 1,024 | RX 9070 XT | 7.0.0        | 2695 µs | 3763 µs |
