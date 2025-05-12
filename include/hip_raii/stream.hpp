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

    operator hipStream_t() const { return stream_; }

    void sync() const {
        hipStreamSynchronize(stream_);
    }

private:
    hipStream_t stream_;
};