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

    // 重載類型轉換運算子，讓 HipStream 可以直接轉換為 hipStream_t
    operator hipStream_t() const { return stream_; }

private:
    hipStream_t stream_;
};