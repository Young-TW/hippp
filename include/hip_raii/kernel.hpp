#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

/// RAII style wrapper to simplify HIP kernel launches.
class HipKernel {
public:
    /// Construct a kernel launcher with configuration parameters.
    /// @param gridDim   Grid dimension used for the launch.
    /// @param blockDim  Block dimension used for the launch.
    /// @param sharedMemBytes Shared memory size in bytes.
    /// @param stream    HIP stream on which the kernel will be launched.
    HipKernel(dim3 gridDim, dim3 blockDim, std::size_t sharedMemBytes = 0,
              hipStream_t stream = nullptr)
        : gridDim_(gridDim),
          blockDim_(blockDim),
          sharedMemBytes_(sharedMemBytes),
          stream_(stream) {}

    /// Launch the specified kernel with the stored configuration.
    /// Any arguments after @p kernel are forwarded to the underlying kernel.
    template <typename KernelFunc, typename... Args>
    void launch(KernelFunc kernel, Args... args) const {
        hipLaunchKernelGGL(kernel, gridDim_, blockDim_, sharedMemBytes_, stream_,
                           args...);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            throw std::runtime_error("hipLaunchKernel failed: " +
                                     std::string(hipGetErrorString(err)));
        }
    }

    /// Getters for configuration parameters.
    dim3 gridDim() const { return gridDim_; }
    dim3 blockDim() const { return blockDim_; }
    std::size_t sharedMemBytes() const { return sharedMemBytes_; }
    hipStream_t stream() const { return stream_; }

    /// Update the stream used for launches.
    void setStream(hipStream_t stream) { stream_ = stream; }

private:
    dim3 gridDim_{};
    dim3 blockDim_{};
    std::size_t sharedMemBytes_{};
    hipStream_t stream_{};
};

