#pragma once

#include <hip/hip_runtime.h>

class HipEvent {
public:
    HipEvent() { hipEventCreate(&event_); }
    ~HipEvent() { hipEventDestroy(event_); }

    operator hipEvent_t() const { return event_; }

private:
    hipEvent_t event_;
};
