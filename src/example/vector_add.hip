#include <hip/hip_runtime.h>
#include <cstdio>
#include <iostream>
#include <iostream>
#include <vector>

#include "hip_raii/buffer.hpp"
#include "hip_raii/stream.hpp"
#include "hip_raii/event.hpp"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

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
    hipMemcpyAsync(a.get(), h_a.data(), size, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(b.get(), h_b.data(), size, hipMemcpyHostToDevice, stream);
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
    hipMemcpyAsync(h_c.data(), c.get(), size, hipMemcpyDeviceToHost, stream);
    stream.sync();

    return 0;
}

int vector_add_origin() {
    constexpr int N = 1 << 20;
    constexpr size_t size = N * sizeof(float);
    constexpr int blockSize = 256;
    constexpr int numBlocks = (N + blockSize - 1) / blockSize;

    // Allocate device memory
    float *a = nullptr, *b = nullptr, *c = nullptr;
    if (hipMalloc(&a, size) != hipSuccess || 
        hipMalloc(&b, size) != hipSuccess || 
        hipMalloc(&c, size) != hipSuccess) {
        std::cerr << "Failed to allocate device memory." << std::endl;
        return -1;
    }

    // Create events and stream
    hipEvent_t start, stop;
    if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess) {
        std::cerr << "Failed to create events." << std::endl;
        hipFree(a);
        hipFree(b);
        hipFree(c);
        return -1;
    }

    hipStream_t stream;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::cerr << "Failed to create stream." << std::endl;
        hipFree(a);
        hipFree(b);
        hipFree(c);
        hipEventDestroy(start);
        hipEventDestroy(stop);
        return -1;
    }

    // Initialize host data
    std::vector<float> h_a(N), h_b(N), h_c(N);
    std::generate(h_a.begin(), h_a.end(), [n = 0]() mutable { return static_cast<float>(n++); });
    std::generate(h_b.begin(), h_b.end(), [n = 0]() mutable { return static_cast<float>(n++); });

    // Copy data from host to device
    hipMemcpyAsync(a, h_a.data(), size, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(b, h_b.data(), size, hipMemcpyHostToDevice, stream);
    hipStreamSynchronize(stream);

    // Warm-up execution
    vector_add<<<numBlocks, blockSize, 0, stream>>>(a, b, c, N);
    hipStreamSynchronize(stream);

    // Measure execution time
    constexpr int runs = 10000;
    float totalMs = 0.0f;

    for (int i = 0; i < runs; ++i) {
        hipEventRecord(start, stream);
        vector_add<<<numBlocks, blockSize, 0, stream>>>(a, b, c, N);
        hipEventRecord(stop, stream);
        hipStreamSynchronize(stream);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, start, stop);
        totalMs += ms;
    }

    std::cout << "Average Origin kernel time over " << runs << " runs: "
              << (totalMs / runs) << " ms\n";

    // Copy results back to host (if needed for verification)
    hipMemcpyAsync(h_c.data(), c, size, hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);

    // Release resources
    hipFree(a);
    hipFree(b);
    hipFree(c);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipStreamDestroy(stream);

    return 0;
}
