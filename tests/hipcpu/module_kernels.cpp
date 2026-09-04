// Kernels loaded at runtime by the HipModule/HipFunction unit test via
// hipModuleLoad. Built into a shared object by tests/hipcpu CMake.
//
// Constraints imposed by the HIP CPU Runtime (these do not exist on GPU
// HIP, where the same wrapper code passes arguments normally):
//   - module-loaded kernels cannot take arguments; data is exchanged
//     through module globals (see HipModule::getGlobal + hipMemcpyToSymbol);
//   - a launched kernel runs once per block with threadIdx == {0,0,0},
//     so blockIdx is the only usable index.
#include <hip/hip_runtime.h>

__device__ const float* scale_in = nullptr;
__device__ float* scale_out = nullptr;

// Each block scales its 64-element slice by two.
extern "C" __global__ void scale_by_two() {
    const int base = blockIdx.x * 64;
    for (int t = 0; t < 64; ++t) {
        scale_out[base + t] = 2.0f * scale_in[base + t];
    }
}

// HIP-CPU's hipModuleLaunchKernel sets the tile context of the loaded
// module through this accessor. It is emitted as a weak symbol only when
// its address is taken, so keep an exported pointer to it.
extern "C" void* (*const hipcpu_tile_ctx_ref)() noexcept =
    &hip::detail::this_tile::_hip_detail_this_tile_so_local_context;
