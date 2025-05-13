# hippp - HIP programming with RAII

hippp is a C++ wrapper for the HIP API, designed to simplify the process of writing GPU code. It provides a RAII (Resource Acquisition Is Initialization) style interface, making it easier to manage resources and ensuring that they are properly released when no longer needed. This library is particularly useful for developers who want to write portable GPU code without dealing with the complexities of the HIP API directly.

## Features

- [x] **Device Memory** (`hipMalloc` / `hipFree`) → `HipBuffer<T>`
- [ ] **Pinned Host Memory** (`hipHostMalloc` / `hipHostFree`) → `HipHostBuffer<T>`
- [x] **Streams** (`hipStreamCreate` / `hipStreamDestroy`) → `HipStream`
- [x] **Events** (`hipEventCreate` / `hipEventDestroy`) → `HipEvent`
- [ ] **Graphs** (`hipGraphCreate` / `hipGraphDestroy`) → `HipGraph`
- [ ] **Graph Executable** (`hipGraphExecCreate` / `hipGraphExecDestroy`) → `HipGraphExec`
- [ ] **Modules & Kernels** (`hipModuleLoad` / `hipModuleUnload`, `hipModuleGetFunction`) → `HipModule`, `HipFunction`
- [ ] **Contexts** (`hipCtxCreate` / `hipCtxDestroy`) → `HipContext`
- [ ] **Arrays** (`hipArrayCreate` / `hipArrayDestroy`, `hipMipmappedArrayCreate` / `hipMipmappedArrayDestroy`) → `HipArray`, `HipMipmappedArray`
- [ ] **Texture Objects** (`hipTexObjectCreate` / `hipTexObjectDestroy`) → `HipTextureObject`
- [ ] **Surface Objects** (`hipSurfObjectCreate` / `hipSurfObjectDestroy`) → `HipSurfaceObject`
- [ ] **Memory Pools** (`hipMemPoolCreate` / `hipMemPoolDestroy`) → `HipMemPool`
- [ ] **External Library Handles** (rocBLAS, rocFFT, rocRAND) → `RocblasHandle`, `RocfftPlan`, `RocrandGenerator`

## Benchmark

| Tasks | Count | GPU | Origin | RAII |
|-------|-------|-----|--------|------|
| Vector Addition | 1,000,000 | RX 7600 XT | 0.0726434 ms | 0.0724291 ms |
