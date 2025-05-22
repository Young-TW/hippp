#include <hip/hip_runtime.h>
#include <vector>

#include "example/device/vector_add.h"

std::vector<float> vector_add_origin(const std::vector<float>& h_a, const std::vector<float>& h_b) {
    if (h_a.size() != h_b.size()) {
        throw std::invalid_argument("Input vectors must have the same size.");
    }

    size_t N = h_a.size();
    size_t size = N * sizeof(float);
    constexpr int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Allocate device memory
    float *a = nullptr, *b = nullptr, *c = nullptr;
    if (hipMalloc(&a, size) != hipSuccess ||
        hipMalloc(&b, size) != hipSuccess ||
        hipMalloc(&c, size) != hipSuccess) {
        throw std::runtime_error("Failed to allocate device memory.");
    }

    // Create stream
    hipStream_t stream;
    if (hipStreamCreate(&stream) != hipSuccess) {
        hipFree(a); hipFree(b); hipFree(c);
        throw std::runtime_error("Failed to create stream.");
    }

    // Copy data from host to device
    hipMemcpyAsync(a, h_a.data(), size, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(b, h_b.data(), size, hipMemcpyHostToDevice, stream);
    hipStreamSynchronize(stream);

    // Launch kernel
    vector_add<<<numBlocks, blockSize, 0, stream>>>(a, b, c, N);
    hipStreamSynchronize(stream);

    // Copy result back to host
    std::vector<float> h_c(N);
    hipMemcpyAsync(h_c.data(), c, size, hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);

    // Release resources
    hipFree(a);
    hipFree(b);
    hipFree(c);
    hipStreamDestroy(stream);

    return h_c;
}