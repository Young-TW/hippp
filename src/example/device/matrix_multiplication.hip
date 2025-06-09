#include <hip/hip_runtime.h>

// C = A * B
// A: M×K, B: K×N, C: M×N
__global__ void matrix_multiplication(const float* A,
                                      const float* B,
                                      float*       C,
                                      int          M,
                                      int          N,
                                      int          K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
