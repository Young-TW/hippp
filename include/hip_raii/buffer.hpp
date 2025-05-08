#pragma once
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstddef>


template <typename T>
class HipBuffer {
public:
    /// @brief Allocate device memory for @p elements elements of type T.
    explicit HipBuffer(std::size_t elements)
        : ptr_(nullptr), elements_(elements) {
        hipError_t err =
            hipMalloc(reinterpret_cast<void**>(&ptr_), elements * sizeof(T));
        if (err != hipSuccess) {
            throw std::runtime_error("hipMalloc failed");
        }
    }

    ~HipBuffer() {
        if (ptr_) {
            hipFree(ptr_);
        }
    }

    HipBuffer(const HipBuffer&)            = delete;
    HipBuffer& operator=(const HipBuffer&) = delete;

    HipBuffer(HipBuffer&& other) noexcept
        : ptr_(other.ptr_), elements_(other.elements_) {
        other.ptr_      = nullptr;
        other.elements_ = 0;
    }

    HipBuffer& operator=(HipBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_)
                hipFree(ptr_);
            ptr_          = other.ptr_;
            elements_     = other.elements_;
            other.ptr_    = nullptr;
            other.elements_ = 0;
        }
        return *this;
    }

    /// Return raw mutable device pointer.
    T* get() { return ptr_; }

    /// Return raw const device pointer.
    const T* get() const { return ptr_; }

    /// @return number of elements stored (not bytes).
    std::size_t size() const { return elements_; }

private:
    T*          ptr_;
    std::size_t elements_;
};
