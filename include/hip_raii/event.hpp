#pragma once

#include <hip/hip_runtime.h>

class HipEvent {
public:
    HipEvent() { hipEventCreate(&event_); }
    ~HipEvent() { hipEventDestroy(event_); }
    hipEvent_t get() const { return event_; }
private:
    hipEvent_t event_;
};
