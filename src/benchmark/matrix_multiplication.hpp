#include <hip/hip_runtime.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "example/device/matrix_multiplication_tiled.h"
#include "example/origin/matrix_multiplication.h"
#include "example/raii/matrix_multiplication.h"

static std::vector<float> run_tiled_benchmark(const std::vector<float>& hA,
                                              const std::vector<float>& hB,
                                              int M, int N, int K) {
    float *dA, *dB, *dC;
    hipMalloc(&dA, M * K * sizeof(float));
    hipMalloc(&dB, K * N * sizeof(float));
    hipMalloc(&dC, M * N * sizeof(float));
    hipMemcpy(dA, hA.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), K * N * sizeof(float), hipMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    hipLaunchKernelGGL(matrix_multiplication_tiled, grid, block, 0, 0, dA, dB,
                       dC, M, N, K);
    hipDeviceSynchronize();

    std::vector<float> hC(M * N);
    hipMemcpy(hC.data(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    return hC;
}

int benchmark_matrix_multiplication(unsigned runs) {
    std::vector<float> h_a(1024 * 1024,
                           1.0f);  // Example size, adjust as needed
    std::vector<float> h_b(1024 * 1024,
                           2.0f);  // Example size, adjust as needed
    int M = 1024;                  // Number of rows in A and C
    int N = 1024;                  // Number of columns in B and C
    int K = 1024;                  // Number of columns in A and rows in B

    // Warm-up runs to ensure the kernel is compiled and ready
    // This is important for accurate benchmarking
    matrix_multiplication_raii(h_a, h_b, M, N, K);
    matrix_multiplication_origin(h_a, h_b, M, N, K);
    run_tiled_benchmark(h_a, h_b, M, N, K);

    long long total_duration_raii = 0, total_duration_origin = 0,
              total_duration_tiled = 0;

    for (unsigned i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> h_c_raii =
            matrix_multiplication_raii(h_a, h_b, M, N, K);
        auto end = std::chrono::high_resolution_clock::now();
        total_duration_raii +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
    }

    for (unsigned i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> h_c_origin =
            matrix_multiplication_origin(h_a, h_b, M, N, K);
        auto end = std::chrono::high_resolution_clock::now();
        total_duration_origin +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
    }

    for (unsigned i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        run_tiled_benchmark(h_a, h_b, M, N, K);
        auto end = std::chrono::high_resolution_clock::now();
        total_duration_tiled +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
    }

    long long avg_origin = total_duration_origin / runs;
    long long avg_tiled = total_duration_tiled / runs;

    std::cout << "[Benchmark] Matrix Multiplication: RAII version took "
              << (total_duration_raii / runs) << " microseconds\n";
    std::cout << "[Benchmark] Matrix Multiplication: Original version took "
              << avg_origin << " microseconds\n";
    std::cout << "[Benchmark] Matrix Multiplication: Tiled version took "
              << avg_tiled << " microseconds\n";
    if (avg_tiled > 0)
        std::cout
            << "[Benchmark] Matrix Multiplication: Tiled speedup vs original: "
            << (static_cast<double>(avg_origin) / avg_tiled) << "x\n";

    return 0;
}
