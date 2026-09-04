#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

/// Stream-ordered memory pool wrapper (hipMallocAsync / hipFreeAsync).
/// Allocations are served from the current device's default memory pool
/// and are ordered with respect to @p stream.
///
/// @note Wraps the default pool only: explicit pool management
///       (hipMemPoolCreate / hipMemPoolDestroy) is not available on the
///       HIP CPU Runtime and therefore not covered by unit tests.
///
/// @note On the HIP CPU Runtime the backing allocation is performed as a
///       task on the stream, so the resulting pointer is written lazily;
///       the caller-provided pointer variable must stay alive until the
///       stream is synchronized (on GPU HIP it is written before
///       hipMallocAsync returns).
class HipMemPool {
public:
    explicit HipMemPool(hipStream_t stream = nullptr) : stream_(stream) {}

    /// @brief Asynchronously allocate @p bytes from the default pool.
    /// @param ptr  set to the new device pointer (see class note about
    ///             when the write happens on the HIP CPU Runtime).
    void allocateAsync(void*& ptr, std::size_t bytes) const {
        hipError_t err = hipMallocAsync(&ptr, bytes, stream_);
        if (err != hipSuccess) {
            throw std::runtime_error("hipMallocAsync failed: " +
                                     std::string(hipGetErrorString(err)));
        }
    }

    /// @brief Asynchronously allocate @p elements objects of type T.
    template <typename T>
    void allocateAsync(T*& ptr, std::size_t elements) const {
        allocateAsync(reinterpret_cast<void*&>(ptr), elements * sizeof(T));
    }

    /// @brief Asynchronously return @p ptr to the pool (stream-ordered).
    void deallocateAsync(void* ptr) const {
        hipError_t err = hipFreeAsync(ptr, stream_);
        if (err != hipSuccess) {
            throw std::runtime_error("hipFreeAsync failed: " +
                                     std::string(hipGetErrorString(err)));
        }
    }

    hipStream_t stream() const { return stream_; }
    void setStream(hipStream_t stream) { stream_ = stream; }

private:
    hipStream_t stream_;
};
