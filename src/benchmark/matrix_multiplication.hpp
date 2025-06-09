#include <iostream>
#include <chrono>
#include <vector>

#include "example/origin/matrix_multiplication.h"
#include "example/raii/matrix_multiplication.h"

int benchmark_matrix_multiplication() {
    std::vector<float> h_a(1024 * 1024, 1.0f); // Example size, adjust as needed
    std::vector<float> h_b(1024 * 1024, 2.0f); // Example size, adjust as needed
    int M = 1024; // Number of rows in A and C
    int N = 1024; // Number of columns in B and C
    int K = 1024; // Number of columns in A and rows in B

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> h_c_raii = matrix_multiplication_raii(h_a, h_b, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_raii = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<float> h_c_origin = matrix_multiplication_origin(h_a, h_b, M, N, K);
    end = std::chrono::high_resolution_clock::now();
    auto duration_origin = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "[Benchmark] Matrix Multiplication: RAII version took " << duration_raii << " microseconds\n";
    std::cout << "[Benchmark] Matrix Multiplication: Original version took " << duration_origin << " microseconds\n";

    return 0;
}