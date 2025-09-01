#pragma once

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <utility>

class HipArray {
public:
    // 1D array: width elements
    explicit HipArray(size_t width, hipChannelFormatDesc desc = hipCreateChannelDesc<float>(), unsigned int flags = hipArrayDefault)
    : array_(nullptr), width_(width), height_(0), desc_(desc) {
        hipError_t err = hipMallocArray(&array_, &desc_, width_, 0, flags);
        if (err != hipSuccess) throw std::runtime_error("hipMallocArray(1D) failed");
    }

    // 2D array: width x height elements
    HipArray(size_t width, size_t height, hipChannelFormatDesc desc = hipCreateChannelDesc<float>(), unsigned int flags = hipArrayDefault)
    : array_(nullptr), width_(width), height_(height), desc_(desc) {
        hipError_t err = hipMallocArray(&array_, &desc_, width_, height_, flags);
        if (err != hipSuccess) throw std::runtime_error("hipMallocArray(2D) failed");
    }

    ~HipArray() {
        if (array_) hipFreeArray(array_);
    }

    // non-copyable
    HipArray(const HipArray&) = delete;
    HipArray& operator=(const HipArray&) = delete;

    // movable
    HipArray(HipArray&& other) noexcept
    : array_(other.array_), width_(other.width_), height_(other.height_), desc_(other.desc_) {
        other.array_ = nullptr;
        other.width_ = other.height_ = 0;
    }

    HipArray& operator=(HipArray&& other) noexcept {
        if (this != &other) {
            if (array_) hipFreeArray(array_);
            array_  = other.array_;
            width_  = other.width_;
            height_ = other.height_;
            desc_   = other.desc_;
            other.array_ = nullptr;
            other.width_ = other.height_ = 0;
        }
        return *this;
    }

    // Upload from host pointer (1D)
    void upload1D(const void* src, size_t count_elems, size_t elem_size = sizeof(float)) {
        if (height_ != 0) throw std::runtime_error("upload1D requires 1D array");
        size_t bytes = count_elems * elem_size;
        hipError_t err = hipMemcpyToArray(array_, 0, 0, src, bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) throw std::runtime_error("hipMemcpyToArray(1D) failed");
    }

    // Upload from host pointer (2D, row-major, tightly packed)
    void upload2D(const void* src, size_t src_pitch_bytes, size_t width_elems, size_t height_rows, size_t elem_size = sizeof(float)) {
        if (height_ == 0) throw std::runtime_error("upload2D requires 2D array");
        size_t width_bytes = width_elems * elem_size;
        hipError_t err = hipMemcpy2DToArray(array_, 0, 0, src, src_pitch_bytes, width_bytes, height_rows, hipMemcpyHostToDevice);
        if (err != hipSuccess) throw std::runtime_error("hipMemcpy2DToArray failed");
    }

    // Download to host pointer (1D)
    void download1D(void* dst, size_t count_elems, size_t elem_size = sizeof(float)) const {
        if (height_ != 0) throw std::runtime_error("download1D requires 1D array");
        size_t bytes = count_elems * elem_size;
        hipError_t err = hipMemcpyFromArray(dst, array_, 0, 0, bytes, hipMemcpyDeviceToHost);
        if (err != hipSuccess) throw std::runtime_error("hipMemcpyFromArray(1D) failed");
    }

    // Download to host pointer (2D)
    void download2D(void* dst, size_t dst_pitch_bytes, size_t width_elems, size_t height_rows, size_t elem_size = sizeof(float)) const {
        if (height_ == 0) throw std::runtime_error("download2D requires 2D array");
        size_t width_bytes = width_elems * elem_size;
        hipError_t err = hipMemcpy2DFromArray(dst, dst_pitch_bytes, array_, 0, 0, width_bytes, height_rows, hipMemcpyDeviceToHost);
        if (err != hipSuccess) throw std::runtime_error("hipMemcpy2DFromArray failed");
    }

    hipArray_t get() const { return array_; }
    size_t width()  const { return width_; }
    size_t height() const { return height_; }
    hipChannelFormatDesc channelDesc() const { return desc_; }

private:
    hipArray_t array_;
    size_t width_;
    size_t height_;
    hipChannelFormatDesc desc_;
};
