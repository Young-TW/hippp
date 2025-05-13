#include <hip/hip_runtime.h>
#include <cstdio>
#include <iostream>
#include <iostream>
#include <vector>

#include "hip_raii/hip_raii.hpp"

__global__ void vector_add(const float* a, const float* b, float* c, int n);

int vector_add_raii() {
    constexpr int N = 1 << 20;
    constexpr size_t size = N * sizeof(float);
    constexpr int blockSize = 256;
    constexpr int numBlocks = (N + blockSize - 1) / blockSize;

    // Use RAII-wrapped resources
    HipBuffer<float> a(N), b(N), c(N);
    HipStream stream;

    // Initialize host data
    std::vector<float> h_a(N), h_b(N), h_c(N);
    std::generate(h_a.begin(), h_a.end(), [n = 0]() mutable { return static_cast<float>(n++); });
    std::generate(h_b.begin(), h_b.end(), [n = 0]() mutable { return static_cast<float>(n++); });

    // Copy data from host to device
    a.copyFromHostAsync(h_a.data(), stream);
    b.copyFromHostAsync(h_b.data(), stream);
    stream.sync();

    // Warm-up execution
    vector_add<<<numBlocks, blockSize, 0, stream>>>(a.get(), b.get(), c.get(), N);
    stream.sync();

    // Measure execution time
    constexpr int runs = 10000;
    float totalMs = 0.0f;

    for (int i = 0; i < runs; ++i) {
        HipEvent start, stop; // Use new events for each measurement
        start.record(stream);
        vector_add<<<numBlocks, blockSize, 0, stream>>>(a.get(), b.get(), c.get(), N);
        stop.record(stream);
        stream.sync();
        totalMs += HipEvent::elapsedTime(start, stop);
    }

    std::cout << "Average RAII kernel time over " << runs << " runs: "
              << (totalMs / runs) << " ms\n";

    // Copy results back to host (if needed for verification)
    c.copyToHostAsync(h_c.data(), stream);
    stream.sync();

    return 0;
}