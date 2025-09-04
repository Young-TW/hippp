#pragma once

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

class HipContext {
public:
    explicit HipContext(int device_id = 0) {
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess) {
            throw std::runtime_error("hipSetDevice failed: " + std::string(hipGetErrorString(err)));
        }

        err = hipCtxCreate(&ctx_, 0, device_id);
        if (err != hipSuccess) {
            throw std::runtime_error("hipCtxCreate failed: " + std::string(hipGetErrorString(err)));
        }
    }

    ~HipContext() {
        if (ctx_) {
            hipCtxDestroy(ctx_);
        }
    }

    // non-copyable
    HipContext(const HipContext&) = delete;
    HipContext& operator=(const HipContext&) = delete;

    // movable
    HipContext(HipContext&& other) noexcept : ctx_(other.ctx_) {
        other.ctx_ = nullptr;
    }

    HipContext& operator=(HipContext&& other) noexcept {
        if (this != &other) {
            if (ctx_) {
                hipCtxDestroy(ctx_);
            }
            ctx_ = other.ctx_;
            other.ctx_ = nullptr;
        }
        return *this;
    }

    hipCtx_t get() const { return ctx_; }

private:
    hipCtx_t ctx_{nullptr};
};
