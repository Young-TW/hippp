// Regression test for GitHub issue #12:
// matrix_multiplication_tiled must produce output matching the naive
// global-memory kernel element-wise within float precision tolerance of 1e-4f.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "example/device/matrix_multiplication.h"
#include "example/device/matrix_multiplication_tiled.h"

static constexpr float TOL = 1e-4f;

static void hip_check(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP error %s at %s:%d\n",
                     hipGetErrorString(err), file, line);
        std::exit(1);
    }
}
#define HIP_CHECK(e) hip_check((e), __FILE__, __LINE__)

static std::vector<float> run_naive(const std::vector<float>& hA,
                                    const std::vector<float>& hB,
                                    int M, int N, int K) {
    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC, M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    hipLaunchKernelGGL(matrix_multiplication, grid, block, 0, 0, dA, dB, dC, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> hC(M * N);
    HIP_CHECK(hipMemcpy(hC.data(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost));
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    return hC;
}

static std::vector<float> run_tiled(const std::vector<float>& hA,
                                    const std::vector<float>& hB,
                                    int M, int N, int K) {
    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC, M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    hipLaunchKernelGGL(matrix_multiplication_tiled, grid, block, 0, 0, dA, dB, dC, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> hC(M * N);
    HIP_CHECK(hipMemcpy(hC.data(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost));
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    return hC;
}

static int check_close(const std::vector<float>& expected,
                       const std::vector<float>& actual,
                       int M, int N, const char* label) {
    int mismatches = 0;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(expected[i] - actual[i]) > TOL) {
            if (mismatches < 5)
                std::fprintf(stderr, "  %s MISMATCH [%d,%d]: expected=%.6f actual=%.6f\n",
                             label, i / N, i % N, expected[i], actual[i]);
            ++mismatches;
        }
    }
    return mismatches;
}

// Test cases: aligned (multiple of TILE_SIZE) and non-aligned sizes.
struct TestCase { int M, N, K; };

static const TestCase CASES[] = {
    {64,  64,  64},   // aligned: no boundary handling needed
    {32,  48,  64},   // non-square, aligned
    {33,  33,  33},   // non-aligned in all dims
    {1,   1,   1},    // smallest possible
    {17,  31,  15},   // non-aligned, odd sizes
};

int main() {
    int failures = 0;

    for (const auto& tc : CASES) {
        int M = tc.M, N = tc.N, K = tc.K;

        std::vector<float> hA(M * K), hB(K * N);
        // Deterministic non-trivial values spanning negatives and positives.
        for (int i = 0; i < M * K; ++i)
            hA[i] = static_cast<float>((i % 7) - 3) * 0.5f;
        for (int i = 0; i < K * N; ++i)
            hB[i] = static_cast<float>((i % 5) - 2) * 0.25f;

        std::vector<float> ref = run_naive(hA, hB, M, N, K);
        std::vector<float> got = run_tiled(hA, hB, M, N, K);

        char label[64];
        std::snprintf(label, sizeof(label), "[%dx%dx%d]", M, N, K);
        int bad = check_close(ref, got, M, N, label);
        if (bad > 0) {
            std::fprintf(stderr, "FAIL %s: %d/%d elements outside tolerance %.1e\n",
                         label, bad, M * N, TOL);
            ++failures;
        } else {
            std::printf("PASS %s\n", label);
        }
    }

    return failures == 0 ? 0 : 1;
}
