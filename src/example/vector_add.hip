#include <hip/hip_runtime.h>
#include <cstdio>
#include <stdexcept>
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
    const int N = 1024;

    HipBuffer<float> a(N), b(N), c(N);
    HipStream stream;
    HipEvent start, stop;

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
    }

    hipMemcpyAsync(a.get(), h_a.data(), N * sizeof(float), hipMemcpyHostToDevice, stream.get());
    hipMemcpyAsync(b.get(), h_b.data(), N * sizeof(float), hipMemcpyHostToDevice, stream.get());

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    hipEventRecord(start.get(), stream.get());
    vector_add<<<numBlocks, blockSize, 0, stream.get()>>>(a.get(), b.get(), c.get(), N);
    hipEventRecord(stop.get(), stream.get());

    hipMemcpyAsync(h_c.data(), c.get(), N * sizeof(float), hipMemcpyDeviceToHost, stream.get());

    hipStreamSynchronize(stream.get());

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start.get(), stop.get());
    printf("Kernel execution time(RAII): %f ms\n", ms);

    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %f + %f != %f\n", i, h_a[i], h_b[i], h_c[i]);
            break;
        }
    }

    return 0;
}


int vector_add_origin() {
    const int N = 1024;
    const int size = N * sizeof(float);

    HipEvent start, stop;

    float *a, *b, *c;
    hipMalloc((void**)&a, size);
    hipMalloc((void**)&b, size);
    hipMalloc((void**)&c, size);

    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
    }

    hipMemcpy(a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(b, h_b, size, hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    hipEventRecord(start.get());
    vector_add<<<numBlocks, blockSize>>>(a, b, c, N);
    hipEventRecord(stop.get());

    hipMemcpy(h_c, c, size, hipMemcpyDeviceToHost);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start.get(), stop.get());
    printf("Kernel execution time(Origin): %f ms\n", ms);

    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %f + %f != %f\n", i, h_a[i], h_b[i], h_c[i]);
            break;
        }
    }

    hipFree(a);
    hipFree(b);
    hipFree(c);

    return 0;
}
