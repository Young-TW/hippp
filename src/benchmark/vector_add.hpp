#include <iostream>
#include <chrono>
#include <vector>

#include "example/origin/vector_add.h"
#include "example/raii/vector_add.h"

int benchmark_vector_add(unsigned runs) {
    std::vector<float> h_a(1 << 20, 1.0f);
    std::vector<float> h_b(1 << 20, 2.0f);

    vector_add_raii(h_a, h_b);
    vector_add_origin(h_a, h_b);

    long long total_duration_raii = 0, total_duration_origin = 0;

    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> h_c_raii = vector_add_raii(h_a, h_b);
        auto end = std::chrono::high_resolution_clock::now();
        total_duration_raii += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> h_c_origin = vector_add_origin(h_a, h_b);
        auto end = std::chrono::high_resolution_clock::now();
        total_duration_origin += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::cout << "[Benchmark] Vector Add: RAII version avg " << (total_duration_raii / runs) << " microseconds\n";
    std::cout << "[Benchmark] Vector Add: Original version avg " << (total_duration_origin / runs) << " microseconds\n";

    return 0;
}