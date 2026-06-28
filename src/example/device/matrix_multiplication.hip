#include <hip/hip_runtime.h>

// C = A * B
// A: M×K, B: K×N, C: M×N
__global__ void matrix_multiplication(const float* A, const float* B, float* C,
                                      int M, int N, int K) {
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

// Tiled SGEMM: loads TILE_SIZE×TILE_SIZE sub-blocks of A and B into shared
// memory so each global-memory value is read once per block instead of once
// per thread, reducing global memory traffic by a factor of TILE_SIZE.
// TILE_SIZE must equal blockDim.x == blockDim.y at launch.
__global__ void matrix_multiplication_tiled(const float* A, const float* B,
                                             float* C, int M, int N, int K) {
    constexpr int TILE_SIZE = 16;
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
