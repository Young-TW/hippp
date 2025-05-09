#pragma once

#include <hip/hip_runtime.h>

class HipStream {
public:
    HipStream() {
        hipStreamCreate(&stream_);
    }
    ~HipStream() {
        hipStreamDestroy(stream_);
    }
    hipStream_t get() const { return stream_; }
private:
    hipStream_t stream_;
};
