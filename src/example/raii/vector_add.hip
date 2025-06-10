#include <hip/hip_runtime.h>

#include <vector>

#include "example/device/vector_add.h"
#include "hip_raii/hip_raii.hpp"

std::vector<float> vector_add_raii(const std::vector<float>& h_a,
                                   const std::vector<float>& h_b) {
    if (h_a.size() != h_b.size()) {
        throw std::invalid_argument("Input vectors must have the same size.");
    }

    size_t N = h_a.size();
    constexpr int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    HipBuffer<float> a(N), b(N), c(N);
    HipStream stream;

    a.copyFromHostAsync(h_a.data(), stream);
    b.copyFromHostAsync(h_b.data(), stream);

    vector_add<<<numBlocks, blockSize, 0, stream>>>(a.get(), b.get(), c.get(),
                                                    N);

    std::vector<float> h_c(N);
    c.copyToHostAsync(h_c.data(), stream);
    stream.sync();

    return h_c;
}