#pragma once

#include <hip/hip_runtime.h>

__global__ void matrix_multiplication(const float* A, const float* B, float* C,
                                      int M, int N, int K);
