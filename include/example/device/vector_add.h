#pragma once

#include <hip/hip_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n);
