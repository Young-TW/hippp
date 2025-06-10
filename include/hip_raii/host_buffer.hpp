#pragma once

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <cstddef>
template <typename T>
class HipHostBuffer {
public:
    /// @brief Allocate host memory for @p elements elements of type T.
    explicit HipHostBuffer(std::size_t elements)
        : ptr_(nullptr), elements_(elements) {
        hipError_t err = hipHostMalloc(reinterpret_cast<void**>(&ptr_), elements * sizeof(T));
        if (err != hipSuccess) {
            throw std::runtime_error("hipHostMalloc failed");
        }
    }

    ~HipHostBuffer() {
        if (ptr_) {
            hipHostFree(ptr_);
        }
    }

    HipHostBuffer(const HipHostBuffer&)            = delete;
    HipHostBuffer& operator=(const HipHostBuffer&) = delete;

    HipHostBuffer(HipHostBuffer&& other) noexcept
        : ptr_(other.ptr_), elements_(other.elements_) {
        other.ptr_      = nullptr;
        other.elements_ = 0;
    }

    HipHostBuffer& operator=(HipHostBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_)
                hipHostFree(ptr_);
            ptr_          = other.ptr_;
            elements_     = other.elements_;
            other.ptr_    = nullptr;
            other.elements_ = 0;
        }
        return *this;
    }

    /// Return raw mutable host pointer.
    T* get() { return ptr_; }

    /// Return raw const host pointer.
    const T* get() const { return ptr_; }

    /// @return number of elements stored (not bytes).
    std::size_t size() const { return elements_; }
    /// @brief Asynchronously copy data from host to device.
    /// @param devicePtr Pointer to the device memory.
    /// @param stream HIP stream to use for the copy (default: nullptr).
    void copyToDeviceAsync(T* devicePtr, hipStream_t stream = nullptr) {
        hipError_t err = hipMemcpyAsync(devicePtr, ptr_, elements_ * sizeof(T),
                                        hipMemcpyHostToDevice, stream);
        if (err != hipSuccess) {
            throw std::runtime_error("hipMemcpyAsync failed");
        }
    }
    /// @brief Asynchronously copy data from device to host.
    /// @param devicePtr Pointer to the device memory.
    /// @param stream HIP stream to use for the copy (default: nullptr).
    void copyFromDeviceAsync(const T* devicePtr, hipStream_t stream = nullptr) {
        hipError_t err = hipMemcpyAsync(ptr_, devicePtr, elements_ * sizeof(T),
                                        hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess) {
            throw std::runtime_error("hipMemcpyAsync failed");
        }
    }
    /// @brief Synchronously copy data from host to device.
    /// @param devicePtr Pointer to the device memory.
    void copyToDeviceSync(T* devicePtr) {
        hipError_t err = hipMemcpy(devicePtr, ptr_, elements_ * sizeof(T),
                                   hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            throw std::runtime_error("hipMemcpy failed");
        }
    }
    /// @brief Synchronously copy data from device to host.
    /// @param devicePtr Pointer to the device memory.
    void copyFromDeviceSync(const T* devicePtr) {
        hipError_t err = hipMemcpy(ptr_, devicePtr, elements_ * sizeof(T),
                                   hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            throw std::runtime_error("hipMemcpy failed");
        }
    }
    /// @brief Fill the host buffer with a specific value.
    /// @param value Value to fill the buffer with.
    void fill(const T& value) {
        for (std::size_t i = 0; i < elements_; ++i) {
            ptr_[i] = value;
        }
    }

private:
    T* ptr_;                ///< Pointer to the allocated host memory.
    std::size_t elements_;  ///< Number of elements allocated.
};
