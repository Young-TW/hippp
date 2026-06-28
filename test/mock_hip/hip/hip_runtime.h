#pragma once
// CPU mock of the HIP runtime for building tests without a ROCm installation.
//
// Kernel correctness is preserved by running each thread in a real std::thread
// so that:
//   - __shared__ (→ static) arrays are genuinely shared within a block
//   - __syncthreads() uses a real std::barrier, acting as a full barrier
//   - threadIdx / blockIdx / blockDim are thread_local, not global state
//
// Blocks execute sequentially; threads within a block execute in parallel.

#include <barrier>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <thread>
#include <vector>

// ── GPU qualifiers ────────────────────────────────────────────────────────────

#define __global__
#define __device__
#define __host__
// Static local: one instance per function scope, shared by all threads that
// call the function concurrently — exactly the semantics of GPU shared memory
// when blocks execute one at a time (which our launcher enforces).
#define __shared__ static

// ── basic types ───────────────────────────────────────────────────────────────

typedef int hipError_t;
static constexpr hipError_t hipSuccess            = 0;
static constexpr hipError_t hipErrorOutOfMemory   = 2;
static constexpr hipError_t hipErrorInvalidValue  = 1;

static constexpr int hipMemcpyHostToDevice   = 1;
static constexpr int hipMemcpyDeviceToHost   = 2;
static constexpr int hipMemcpyDeviceToDevice = 3;

struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned x=1, unsigned y=1, unsigned z=1)
        : x(x), y(y), z(z) {}
};

// ── per-thread state ──────────────────────────────────────────────────────────

namespace hip_mock {
    struct idx3 { int x, y, z; };
    inline thread_local idx3 tls_threadIdx{0, 0, 0};
    inline thread_local idx3 tls_blockIdx {0, 0, 0};
    inline thread_local idx3 tls_blockDim {1, 1, 1};
    inline thread_local idx3 tls_gridDim  {1, 1, 1};
    inline thread_local std::barrier<>* tls_bar = nullptr;
} // namespace hip_mock

#define threadIdx (hip_mock::tls_threadIdx)
#define blockIdx  (hip_mock::tls_blockIdx)
#define blockDim  (hip_mock::tls_blockDim)
#define gridDim   (hip_mock::tls_gridDim)

inline void __syncthreads() {
    if (hip_mock::tls_bar) hip_mock::tls_bar->arrive_and_wait();
}

// ── memory management ─────────────────────────────────────────────────────────

template<typename T>
inline hipError_t hipMalloc(T** ptr, std::size_t sz) {
    if (!ptr || !sz) return hipErrorInvalidValue;
    *ptr = static_cast<T*>(std::malloc(sz));
    return *ptr ? hipSuccess : hipErrorOutOfMemory;
}
inline hipError_t hipFree(void* ptr) {
    std::free(ptr); return hipSuccess;
}
inline hipError_t hipMemcpy(void* dst, const void* src, std::size_t n, int) {
    std::memcpy(dst, src, n); return hipSuccess;
}
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }

inline const char* hipGetErrorString(hipError_t err) {
    switch (err) {
        case hipSuccess:          return "hipSuccess";
        case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
        case hipErrorInvalidValue:return "hipErrorInvalidValue";
        default:                  return "(unknown hipError_t)";
    }
}

// ── kernel launcher ───────────────────────────────────────────────────────────
// Launches blocks sequentially; each block spawns one std::thread per GPU
// thread so shared memory and barriers behave correctly.

template<typename Kernel, typename... Args>
void hipLaunchKernelImpl(Kernel kern, dim3 grid, dim3 block,
                          std::size_t /*sharedMem*/, void* /*stream*/,
                          Args... args) {
    const int nthreads = static_cast<int>(block.x * block.y * block.z);
    for (int bz = 0; bz < static_cast<int>(grid.z); ++bz)
    for (int by = 0; by < static_cast<int>(grid.y); ++by)
    for (int bx = 0; bx < static_cast<int>(grid.x); ++bx) {
        std::barrier<> bar(nthreads);
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(nthreads));
        for (int tz = 0; tz < static_cast<int>(block.z); ++tz)
        for (int ty = 0; ty < static_cast<int>(block.y); ++ty)
        for (int tx = 0; tx < static_cast<int>(block.x); ++tx) {
            threads.emplace_back([=, &bar]() mutable {
                hip_mock::tls_threadIdx = {tx, ty, tz};
                hip_mock::tls_blockIdx  = {bx, by, bz};
                hip_mock::tls_blockDim  = {static_cast<int>(block.x),
                                           static_cast<int>(block.y),
                                           static_cast<int>(block.z)};
                hip_mock::tls_gridDim   = {static_cast<int>(grid.x),
                                           static_cast<int>(grid.y),
                                           static_cast<int>(grid.z)};
                hip_mock::tls_bar       = &bar;
                kern(args...);
            });
        }
        for (auto& t : threads) t.join();
    }
}

#define hipLaunchKernelGGL(kern, grid, block, sharedMem, stream, ...) \
    hipLaunchKernelImpl((kern), (grid), (block), \
                        static_cast<std::size_t>(sharedMem), (stream), \
                        __VA_ARGS__)
