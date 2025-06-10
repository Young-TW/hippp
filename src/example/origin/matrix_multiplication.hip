#include <hip/hip_runtime.h>

#include <vector>

#include "example/device/matrix_multiplication.h"

std::vector<float> matrix_multiplication_origin(const std::vector<float>& h_A,
                                                const std::vector<float>& h_B,
                                                int M, int N, int K) {
    if (h_A.size() != M * K || h_B.size() != K * N) {
        throw std::invalid_argument(
            "Input matrices must have compatible sizes.");
    }

    size_t size_A = h_A.size() * sizeof(float);
    size_t size_B = h_B.size() * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    constexpr int blockSize = 16;  // Assuming square blocks for simplicity
    dim3 block(blockSize, blockSize);
    dim3 grid((N + blockSize - 1) / blockSize, (M + blockSize - 1) / blockSize);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (hipMalloc(&d_A, size_A) != hipSuccess ||
        hipMalloc(&d_B, size_B) != hipSuccess ||
        hipMalloc(&d_C, size_C) != hipSuccess) {
        throw std::runtime_error("Failed to allocate device memory.");
    }
    try {
        // Copy data from host to device
        hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice);
        // Launch kernel
        matrix_multiplication<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            hipFree(d_A);
            hipFree(d_B);
            hipFree(d_C);
            throw std::runtime_error("Kernel launch failed: " +
                                     std::string(hipGetErrorString(err)));
        }

        // Copy result back to host
        std::vector<float> h_C(M * N);
        hipMemcpy(h_C.data(), d_C, size_C, hipMemcpyDeviceToHost);
        // Release resources
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
        return h_C;
    } catch (...) {
        // Ensure resources are released in case of an exception
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
        throw;  // Re-throw the exception after cleanup
    }
    return {};  // This line is never reached, but added to satisfy the compiler
}
