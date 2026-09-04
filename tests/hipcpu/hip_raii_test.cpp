// Unit tests for hippp RAII wrappers running on AMD's official HIP CPU
// Runtime (ROCm/HIP-CPU) — no GPU required.
//
// The tests intentionally include only the wrappers whose HIP APIs are
// implemented by HIP-CPU: HipBuffer, HipHostBuffer, HipStream, HipEvent,
// HipKernel. (HipArray/HipTexture/... require driver-level APIs that
// HIP-CPU does not provide.)

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <utility>
#include <vector>

#include "example/device/vector_add.h"
#include "hip_raii/buffer.hpp"
#include "hip_raii/event.hpp"
#include "hip_raii/host_buffer.hpp"
#include "hip_raii/kernel.hpp"
#include "hip_raii/stream.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            ++g_failures;                                            \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__,    \
                        #cond);                                      \
        }                                                            \
    } while (0)

void test_buffer_basics() {
    std::puts("[test] HipBuffer basics");
    constexpr std::size_t N = 1024;

    HipBuffer<float> buf(N);
    CHECK(buf.size() == N);
    CHECK(buf.get() != nullptr);

    // Move construction transfers ownership.
    HipBuffer<float> moved(std::move(buf));
    CHECK(moved.size() == N);
    CHECK(moved.get() != nullptr);
    CHECK(buf.get() == nullptr);
    CHECK(buf.size() == 0);

    // Move assignment.
    HipBuffer<float> other(16);
    other = std::move(moved);
    CHECK(other.size() == N);
    CHECK(moved.get() == nullptr);
}

void test_buffer_copy_roundtrip() {
    std::puts("[test] HipBuffer H2D/D2H roundtrip");
    constexpr std::size_t N = 4096;

    std::vector<float> h_in(N), h_out(N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    HipBuffer<float> d(N);
    d.copyFromHostAsync(h_in.data());
    d.copyToHostAsync(h_out.data());
    hipDeviceSynchronize();

    for (std::size_t i = 0; i < N; ++i) CHECK(h_out[i] == h_in[i]);
}

void test_host_buffer() {
    std::puts("[test] HipHostBuffer");
    constexpr std::size_t N = 512;

    HipHostBuffer<int> host(N);
    CHECK(host.size() == N);
    CHECK(host.get() != nullptr);
    host.fill(7);
    for (std::size_t i = 0; i < N; ++i) CHECK(host.get()[i] == 7);

    // Move construction.
    HipHostBuffer<int> moved(std::move(host));
    CHECK(moved.size() == N);
    CHECK(host.get() == nullptr);

    // Roundtrip through a device buffer.
    HipBuffer<int> dev(N);
    moved.copyToDeviceSync(dev.get());
    HipHostBuffer<int> back(N);
    back.copyFromDeviceSync(dev.get());
    for (std::size_t i = 0; i < N; ++i) CHECK(back.get()[i] == 7);
}

void test_stream() {
    std::puts("[test] HipStream");
    HipStream stream;
    CHECK(static_cast<hipStream_t>(stream) != nullptr);

    constexpr std::size_t N = 2048;
    std::vector<float> h(N, 1.5f), out(N, 0.0f);
    HipBuffer<float> d(N);
    d.copyFromHostAsync(h.data(), stream);
    d.copyToHostAsync(out.data(), stream);
    stream.sync();
    for (std::size_t i = 0; i < N; ++i) CHECK(out[i] == h[i]);
}

void test_event() {
    std::puts("[test] HipEvent");
    HipEvent start, stop;

    HipStream stream;
    constexpr std::size_t N = 1 << 20;
    std::vector<float> h(N, 1.0f);
    HipBuffer<float> d(N);

    start.record(stream);
    d.copyFromHostAsync(h.data(), stream);
    stop.record(stream);
    stream.sync();

    const float ms = HipEvent::elapsedTime(start, stop);
    CHECK(std::isfinite(ms));
    CHECK(ms >= 0.0f);
    std::printf("  elapsed time recorded: %.3f ms\n", ms);
}

void test_kernel_vector_add() {
    std::puts("[test] HipKernel vector_add (device kernel from src/example/device)");
    constexpr std::size_t N = 1'048'576;  // same size as the project benchmark
    constexpr int blockSize = 256;
    const int numBlocks = static_cast<int>((N + blockSize - 1) / blockSize);

    std::vector<float> h_a(N), h_b(N), h_c(N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i % 1000);
        h_b[i] = 2.0f * static_cast<float>(i % 1000);
    }

    HipBuffer<float> a(N), b(N), c(N);
    HipStream stream;
    a.copyFromHostAsync(h_a.data(), stream);
    b.copyFromHostAsync(h_b.data(), stream);

    HipKernel kernel(dim3(numBlocks), dim3(blockSize), 0, stream);
    CHECK(kernel.gridDim().x == static_cast<unsigned>(numBlocks));
    CHECK(kernel.blockDim().x == static_cast<unsigned>(blockSize));
    kernel.launch(vector_add, a.get(), b.get(), c.get(), static_cast<int>(N));

    c.copyToHostAsync(h_c.data(), stream);
    stream.sync();

    for (std::size_t i = 0; i < N; ++i) CHECK(h_c[i] == h_a[i] + h_b[i]);
}

void test_end_to_end_raii_flow() {
    std::puts("[test] end-to-end RAII flow (mirrors src/example/raii/vector_add.hip)");
    constexpr std::size_t N = 100'000;
    std::vector<float> h_a(N, 3.0f), h_b(N, 4.0f), h_c(N);

    constexpr int blockSize = 256;
    const int numBlocks = static_cast<int>((N + blockSize - 1) / blockSize);

    HipBuffer<float> a(N), b(N), c(N);
    HipStream stream;
    a.copyFromHostAsync(h_a.data(), stream);
    b.copyFromHostAsync(h_b.data(), stream);

    HipKernel kernel(dim3(numBlocks), dim3(blockSize), 0, stream);
    kernel.launch(vector_add, a.get(), b.get(), c.get(), static_cast<int>(N));

    c.copyToHostAsync(h_c.data(), stream);
    stream.sync();

    for (std::size_t i = 0; i < N; ++i) CHECK(h_c[i] == 7.0f);
}

}  // namespace

int main() {
    test_buffer_basics();
    test_buffer_copy_roundtrip();
    test_host_buffer();
    test_stream();
    test_event();
    test_kernel_vector_add();
    test_end_to_end_raii_flow();

    if (g_failures == 0) {
        std::puts("ALL TESTS PASSED");
        return 0;
    }
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
}
