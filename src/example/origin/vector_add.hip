#include <hip/hip_runtime.h>
#include <vector>
#include <stdexcept>

#include "example/device/vector_add.h"

std::vector<float> vector_add_origin(const std::vector<float>& h_a, const std::vector<float>& h_b) {
    if (h_a.size() != h_b.size()) {
        throw std::invalid_argument("Input vectors must have the same size.");
    }

    size_t N = h_a.size();
    size_t size = N * sizeof(float);
    constexpr int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    float *a = nullptr, *b = nullptr, *c = nullptr;
    hipError_t err;

    if ((err = hipMalloc(&a, size)) != hipSuccess ||
        (err = hipMalloc(&b, size)) != hipSuccess ||
        (err = hipMalloc(&c, size)) != hipSuccess) {
        if (a) hipFree(a);
        if (b) hipFree(b);
        if (c) hipFree(c);
        throw std::runtime_error("Failed to allocate device memory.");
    }

    hipStream_t stream;
    if ((err = hipStreamCreate(&stream)) != hipSuccess) {
        hipFree(a); hipFree(b); hipFree(c);
        throw std::runtime_error("Failed to create stream.");
    }

    if ((err = hipMemcpyAsync(a, h_a.data(), size, hipMemcpyHostToDevice, stream)) != hipSuccess ||
        (err = hipMemcpyAsync(b, h_b.data(), size, hipMemcpyHostToDevice, stream)) != hipSuccess) {
        hipFree(a); hipFree(b); hipFree(c); hipStreamDestroy(stream);
        throw std::runtime_error("Failed to copy data to device.");
    }

    vector_add<<<numBlocks, blockSize, 0, stream>>>(a, b, c, N);

    if ((err = hipGetLastError()) != hipSuccess) {
        hipFree(a); hipFree(b); hipFree(c); hipStreamDestroy(stream);
        throw std::runtime_error("Kernel launch failed.");
    }

    std::vector<float> h_c(N);
    if ((err = hipMemcpyAsync(h_c.data(), c, size, hipMemcpyDeviceToHost, stream)) != hipSuccess) {
        hipFree(a); hipFree(b); hipFree(c); hipStreamDestroy(stream);
        throw std::runtime_error("Failed to copy result to host.");
    }

    if ((err = hipStreamSynchronize(stream)) != hipSuccess) {
        hipFree(a); hipFree(b); hipFree(c); hipStreamDestroy(stream);
        throw std::runtime_error("Stream synchronize failed.");
    }

    hipFree(a);
    hipFree(b);
    hipFree(c);
    hipStreamDestroy(stream);

    return h_c;
}