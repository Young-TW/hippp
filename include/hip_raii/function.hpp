#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

/// Lightweight, non-owning wrapper around a hipFunction_t obtained from a
/// HipModule. Copyable (the handle is owned by the parent HipModule).
class HipFunction {
public:
    HipFunction() = default;
    explicit HipFunction(hipFunction_t fn) : fn_(fn) {}

    /// Launch with no kernel arguments via the HIP_LAUNCH_PARAM extra
    /// configuration. Works on both GPU HIP and the HIP CPU Runtime.
    /// @note HIP-CPU executes module-loaded kernels with one fiber per
    ///       block (threadIdx is always {0,0,0}), and does not support
    ///       kernel arguments — exchange data through module globals.
    void launch(dim3 gridDim, dim3 blockDim, unsigned int sharedMemBytes = 0,
                hipStream_t stream = nullptr) const {
        void* args = nullptr;
        std::size_t argsSize = 0;
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, &argsSize,
                          HIP_LAUNCH_PARAM_END};
        hipError_t err = hipModuleLaunchKernel(
            fn_, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
            blockDim.z, sharedMemBytes, stream, nullptr, config);
        if (err != hipSuccess) {
            throw std::runtime_error("hipModuleLaunchKernel failed: " +
                                     std::string(hipGetErrorString(err)));
        }
    }

    /// Launch with kernel arguments via a kernel_params array.
    /// @note Not supported by the HIP CPU Runtime (it has no device side
    ///       to marshal arguments to, and rejects any non-empty params).
    template <typename... Args>
    void launch(dim3 gridDim, dim3 blockDim, unsigned int sharedMemBytes,
                hipStream_t stream, Args&... args) const {
        void* params[] = {&args...};
        hipError_t err = hipModuleLaunchKernel(
            fn_, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
            blockDim.z, sharedMemBytes, stream, params, nullptr);
        if (err != hipSuccess) {
            throw std::runtime_error("hipModuleLaunchKernel failed: " +
                                     std::string(hipGetErrorString(err)));
        }
    }

    hipFunction_t get() const { return fn_; }
    explicit operator bool() const { return fn_ != nullptr; }

private:
    hipFunction_t fn_{nullptr};
};
