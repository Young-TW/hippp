#include <hip/hip_runtime.h>

#include <vector>

#include "example/device/matrix_multiplication.h"
#include "hip_raii/hip_raii.hpp"

std::vector<float> matrix_multiplication_raii(const std::vector<float>& h_A,
                                              const std::vector<float>& h_B,
                                              int M, int N, int K) {
    if (h_A.size() != M * K || h_B.size() != K * N) {
        throw std::invalid_argument(
            "Input matrices must have compatible sizes.");
    }

    constexpr int blockSize = 16;  // Assuming square blocks for simplicity
    dim3 block(blockSize, blockSize);
    dim3 grid((N + blockSize - 1) / blockSize, (M + blockSize - 1) / blockSize);
    // use RAII to manage device memory
    HipBuffer<float> a(M * K), b(K * N), c(M * N);
    HipStream stream;
    a.copyFromHostAsync(h_A.data(), stream);
    b.copyFromHostAsync(h_B.data(), stream);
    HipKernel kernel(grid, block, 0, stream);
    kernel.launch(matrix_multiplication, a.get(), b.get(), c.get(), M, N, K);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(hipGetErrorString(err)));
    }

    std::vector<float> h_C(M * N);
    c.copyToHostAsync(h_C.data(), stream);
    stream.sync();  // Ensure all operations are complete before returning
    return h_C;
    // Note: RAII will automatically free device memory when 'a', 'b', and 'c'
    // go out of scope
}
