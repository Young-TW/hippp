#pragma once

#include <hip/hip_runtime.h>
#include <stdexcept>

class HipEvent {
public:
    HipEvent() {
        hipError_t err = hipEventCreate(&event_);
        if (err != hipSuccess) {
            throw std::runtime_error("hipEventCreate failed: " + std::string(hipGetErrorString(err)));
        }
    }

    ~HipEvent() {
        hipError_t err = hipEventDestroy(event_);
        if (err != hipSuccess) {
            // Log or handle the error (destructor should not throw exceptions)
        }
    }

    // Record the event on a specific stream
    void record(hipStream_t stream = nullptr) {
        hipError_t err = hipEventRecord(event_, stream);
        if (err != hipSuccess) {
            throw std::runtime_error("hipEventRecord failed: " + std::string(hipGetErrorString(err)));
        }
    }

    // Calculate elapsed time between two events
    static float elapsedTime(const HipEvent& start, const HipEvent& stop) {
        float ms = 0.0f;
        hipError_t err = hipEventElapsedTime(&ms, start.event_, stop.event_);
        if (err != hipSuccess) {
            throw std::runtime_error("hipEventElapsedTime failed: " + std::string(hipGetErrorString(err)));
        }
        return ms;
    }

    // Implicit conversion to hipEvent_t
    operator hipEvent_t() const { return event_; }

private:
    hipEvent_t event_;
};
