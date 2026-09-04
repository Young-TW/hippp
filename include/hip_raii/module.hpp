#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "hip_raii/function.hpp"

/// RAII style wrapper for a HIP module loaded with hipModuleLoad.
/// On GPU HIP the file is a code object (.hsaco/.co); the HIP CPU Runtime
/// loads a shared object (.so/.dll) instead.
class HipModule {
public:
    /// @brief Load the module from @p path (device code object / shared lib).
    explicit HipModule(const char* path) {
        hipError_t err = hipModuleLoad(&module_, path);
        if (err != hipSuccess) {
            throw std::runtime_error("hipModuleLoad failed: " +
                                     std::string(hipGetErrorString(err)));
        }
    }

    ~HipModule() {
#if defined(__HIP_CPU_RT__)
        // HIP-CPU does not expose hipModuleUnload; loaded shared objects
        // are released by the OS at process exit.
#else
        if (module_) {
            hipModuleUnload(module_);
        }
#endif
    }

    HipModule(const HipModule&) = delete;
    HipModule& operator=(const HipModule&) = delete;

    HipModule(HipModule&& other) noexcept : module_(other.module_) {
        other.module_ = nullptr;
    }

    HipModule& operator=(HipModule&& other) noexcept {
        if (this != &other) {
#if !defined(__HIP_CPU_RT__)
            if (module_) hipModuleUnload(module_);
#endif
            module_ = other.module_;
            other.module_ = nullptr;
        }
        return *this;
    }

    /// @return the function named @p name inside the module.
    HipFunction getFunction(const char* name) const {
        hipFunction_t fn = nullptr;
        hipError_t err = hipModuleGetFunction(&fn, module_, name);
        if (err != hipSuccess) {
            throw std::runtime_error("hipModuleGetFunction failed: " +
                                     std::string(hipGetErrorString(err)));
        }
        return HipFunction{fn};
    }

    /// @return device-side address of the module global named @p name
    ///         (use hipMemcpyToSymbol/FromSymbol to read or modify it).
    void* getGlobal(const char* name, std::size_t* sizeBytes = nullptr) const {
        hipDeviceptr_t ptr = nullptr;
        std::size_t sz = 0;
        hipError_t err = hipModuleGetGlobal(&ptr, &sz, module_, name);
        if (err != hipSuccess) {
            throw std::runtime_error("hipModuleGetGlobal failed: " +
                                     std::string(hipGetErrorString(err)));
        }
        if (sizeBytes) *sizeBytes = sz;
        return ptr;
    }

    hipModule_t get() const { return module_; }
    explicit operator bool() const { return module_ != nullptr; }

private:
    hipModule_t module_{nullptr};
};
