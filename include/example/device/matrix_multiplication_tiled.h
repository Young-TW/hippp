#pragma once

#include <hip/hip_runtime.h>

// Tiled SGEMM kernel. Uses TILE_SIZE=16 shared-memory tiles (compile-time
// constexpr inside the kernel) so callers can tune it by changing that
// constant and recompiling without any API changes.
// Must be launched with block size 16×16.
__global__ void matrix_multiplication_tiled(const float* A, const float* B,
                                             float* C, int M, int N, int K);
