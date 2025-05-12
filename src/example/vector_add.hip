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
    const int N = 1 << 20;
    const size_t size = N * sizeof(float);
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    HipBuffer<float> a(N), b(N), c(N);
    HipStream stream;
    HipEvent start, stop;

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
    }
    hipMemcpyAsync(a.get(), h_a.data(), size, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(b.get(), h_b.data(), size, hipMemcpyHostToDevice, stream);
    stream.sync();

    // Warm-up
    hipDeviceSynchronize();
    vector_add<<<numBlocks, blockSize, 0, stream>>>(a.get(), b.get(), c.get(), N);
    stream.sync();

    // Measurement
    const int runs = 10000;
    float totalMs = 0.0f;
    for (int i = 0; i < runs; ++i) {
        hipEventRecord(start, stream);
        vector_add<<<numBlocks, blockSize, 0, stream>>>(a.get(), b.get(), c.get(), N);
        hipEventRecord(stop, stream);
        stream.sync();
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start, stop);
        totalMs += ms;
    }
    std::cout << "Average RAII kernel time over " << runs << " runs: "
              << (totalMs / runs) << " ms\n";
    return 0;
}

int vector_add_origin() {
    const int N = 1 << 20;
    const size_t size = N * sizeof(float);
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    float *a, *b, *c;
    hipMalloc((void**)&a, size);
    hipMalloc((void**)&b, size);
    hipMalloc((void**)&c, size);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
    }

    hipMemcpy(a, h_a.data(), size, hipMemcpyHostToDevice);
    hipMemcpy(b, h_b.data(), size, hipMemcpyHostToDevice);

    hipDeviceSynchronize();
    vector_add<<<numBlocks, blockSize>>>(a, b, c, N);
    hipDeviceSynchronize();

    const int runs = 10000;
    float totalMs = 0.0f;
    for (int i = 0; i < runs; ++i) {
        hipEventRecord(start, 0);
        vector_add<<<numBlocks, blockSize>>>(a, b, c, N);
        hipEventRecord(stop, 0);
        hipDeviceSynchronize();
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start, stop);
        totalMs += ms;
    }

    std::cout << "Average Origin kernel time over " << runs << " runs: "
              << (totalMs / runs) << " ms\n";

    // 釋放資源
    hipFree(a);
    hipFree(b);
    hipFree(c);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}