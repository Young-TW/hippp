#include <iostream>
#include <chrono>
#include <vector>

#include "example/origin/vector_add.h"
#include "example/raii/vector_add.h"

int benchmark_vector_add() {
    std::vector<float> h_a(1024, 1.0f);
    std::vector<float> h_b(1024, 2.0f);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> h_c_raii = vector_add_raii(h_a, h_b);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_raii = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    std::vector<float> h_c_origin = vector_add_origin(h_a, h_b);
    end = std::chrono::high_resolution_clock::now();
    auto duration_origin = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "[Benchmark] Vector Add: RAII version took " << duration_raii << " microseconds\n";
    std::cout << "[Benchmark] Vector Add: Original version took " << duration_origin << " microseconds\n";

    return 0;
}