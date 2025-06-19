#pragma once

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

/// RAII wrapper for a HIP kernel function obtained from a module.
class HipKernel {
public:
    /// Construct an empty kernel handle.
    HipKernel() : function_(nullptr) {}

    /// Load a kernel function from a HIP module.
    HipKernel(hipModule_t module, const char* name) { load(module, name); }

    /// Load a kernel function from a HIP module.
    void load(hipModule_t module, const char* name) {
        hipError_t err = hipModuleGetFunction(&function_, module, name);
        if (err != hipSuccess) {
            throw std::runtime_error("hipModuleGetFunction failed: " +
                                     std::string(hipGetErrorString(err)));
        }
    }

    /// Launch the kernel with the given parameters.
    template <typename... Args>
    void launch(dim3 gridDim, dim3 blockDim, size_t sharedMemBytes,
                hipStream_t stream, Args... args) const {
        void* params[] = { reinterpret_cast<void*>(&args)... };
        hipError_t err = hipModuleLaunchKernel(
            function_, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
            blockDim.z, sharedMemBytes, stream, params, nullptr);
        if (err != hipSuccess) {
            throw std::runtime_error("hipModuleLaunchKernel failed: " +
                                     std::string(hipGetErrorString(err)));
        }
    }

    /// Launch the kernel with default shared memory of 0 and stream nullptr.
    template <typename... Args>
    void launch(dim3 gridDim, dim3 blockDim, Args... args) const {
        launch(gridDim, blockDim, 0, nullptr, args...);
    }

    /// Return underlying hipFunction_t handle.
    hipFunction_t get() const { return function_; }

private:
    hipFunction_t function_;
};

