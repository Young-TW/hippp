#pragma once

// Mock HIP runtime header for testing without ROCm installation
#include <cstddef>
#include <cstdlib>
#include <cstring>

// HIP error codes
typedef enum {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorNotInitialized = 3,
    hipErrorInitializationError = 4,
    hipErrorDeviceNotFound = 100,
    hipErrorInvalidDevice = 101
} hipError_t;

// HIP types
typedef void* hipCtx_t;
typedef void* hipStream_t;
typedef void* hipModule_t;
typedef void* hipFunction_t;
typedef void* hipEvent_t;
typedef void* hipArray_t;

// Channel format descriptors
struct hipChannelFormatDesc {
    int x, y, z, w;
    int f;
    
    hipChannelFormatDesc(int x = 8, int y = 0, int z = 0, int w = 0, int f = 0)
        : x(x), y(y), z(z), w(w), f(f) {}
};

// Channel format helper
template<typename T>
inline hipChannelFormatDesc hipCreateChannelDesc() {
    return hipChannelFormatDesc(8 * sizeof(T), 0, 0, 0, 0);
}

// dim3 structure for kernel dimensions
struct dim3 {
    unsigned int x, y, z;
    
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) 
        : x(x), y(y), z(z) {}
};

// Memory copy types
typedef enum {
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

// Host memory flags
#define hipHostMallocDefault 0
#define hipHostMallocPortable 1
#define hipHostMallocMapped 2
#define hipHostMallocWriteCombined 4

// Array flags
#define hipArrayDefault 0

// Basic HIP functions
inline hipError_t hipSetDevice(int device) {
    return (device >= 0) ? hipSuccess : hipErrorInvalidDevice;
}

inline hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, int device) {
    if (ctx && device >= 0) {
        *ctx = reinterpret_cast<hipCtx_t>(0x1234); // Mock context pointer
        return hipSuccess;
    }
    return hipErrorInvalidValue;
}

inline hipError_t hipCtxDestroy(hipCtx_t ctx) {
    return ctx ? hipSuccess : hipErrorInvalidValue;
}

inline hipError_t hipMalloc(void** ptr, size_t size) {
    if (ptr && size > 0) {
        *ptr = malloc(size); // Use regular malloc for testing
        return *ptr ? hipSuccess : hipErrorOutOfMemory;
    }
    return hipErrorInvalidValue;
}

inline hipError_t hipFree(void* ptr) {
    if (ptr) {
        free(ptr);
        return hipSuccess;
    }
    return hipErrorInvalidValue;
}

inline hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags = hipHostMallocDefault) {
    if (ptr && size > 0) {
        *ptr = malloc(size); // Use regular malloc for testing
        return *ptr ? hipSuccess : hipErrorOutOfMemory;
    }
    return hipErrorInvalidValue;
}

inline hipError_t hipHostFree(void* ptr) {
    if (ptr) {
        free(ptr);
        return hipSuccess;
    }
    return hipErrorInvalidValue;
}

// Array functions
inline hipError_t hipMallocArray(hipArray_t* array, const hipChannelFormatDesc* desc, 
                                size_t width, size_t height, unsigned int flags) {
    if (array) {
        *array = malloc(width * height * 4); // Mock allocation
        return *array ? hipSuccess : hipErrorOutOfMemory;
    }
    return hipErrorInvalidValue;
}

inline hipError_t hipFreeArray(hipArray_t array) {
    if (array) {
        free(array);
        return hipSuccess;
    }
    return hipErrorInvalidValue;
}

inline hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset,
                                  const void* src, size_t count, hipMemcpyKind kind) {
    if (dst && src && count > 0) {
        memcpy(dst, src, count);
        return hipSuccess;
    }
    return hipErrorInvalidValue;
}

inline hipError_t hipMemcpyFromArray(void* dst, hipArray_t src, size_t wOffset, 
                                    size_t hOffset, size_t count, hipMemcpyKind kind) {
    if (dst && src && count > 0) {
        memcpy(dst, src, count);
        return hipSuccess;
    }
    return hipErrorInvalidValue;
}

inline hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset,
                                     const void* src, size_t spitch, size_t width,
                                     size_t height, hipMemcpyKind kind) {
    return hipSuccess; // Mock implementation
}

inline hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_t src,
                                       size_t wOffset, size_t hOffset, size_t width,
                                       size_t height, hipMemcpyKind kind) {
    return hipSuccess; // Mock implementation
}

inline hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, 
                                 hipMemcpyKind kind, hipStream_t stream) {
    if (dst && src && sizeBytes > 0) {
        memcpy(dst, src, sizeBytes);
        return hipSuccess;
    }
    return hipErrorInvalidValue;
}

// Stream functions
inline hipError_t hipStreamCreate(hipStream_t* stream) {
    if (stream) {
        *stream = reinterpret_cast<hipStream_t>(0x5678);
        return hipSuccess;
    }
    return hipErrorInvalidValue;
}

inline hipError_t hipStreamDestroy(hipStream_t stream) {
    return stream ? hipSuccess : hipErrorInvalidValue;
}

inline hipError_t hipStreamSynchronize(hipStream_t stream) {
    return hipSuccess; // Mock synchronization always succeeds
}

// Event functions
inline hipError_t hipEventCreate(hipEvent_t* event) {
    if (event) {
        *event = reinterpret_cast<hipEvent_t>(0x9ABC);
        return hipSuccess;
    }
    return hipErrorInvalidValue;
}

inline hipError_t hipEventDestroy(hipEvent_t event) {
    return event ? hipSuccess : hipErrorInvalidValue;
}

inline hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = nullptr) {
    return hipSuccess; // Mock always succeeds
}

inline hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    if (ms) {
        *ms = 1.0f; // Mock elapsed time
        return hipSuccess;
    }
    return hipErrorInvalidValue;
}

// Error handling
inline hipError_t hipGetLastError() {
    return hipSuccess; // Mock always returns success
}

inline const char* hipGetErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess: return "hipSuccess";
        case hipErrorInvalidValue: return "hipErrorInvalidValue";
        case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
        case hipErrorNotInitialized: return "hipErrorNotInitialized";
        case hipErrorInitializationError: return "hipErrorInitializationError";
        case hipErrorDeviceNotFound: return "hipErrorDeviceNotFound";
        case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
        default: return "hipErrorUnknown";
    }
}

// Kernel launch function (mock)
template<typename KernelFunc, typename... Args>
inline hipError_t hipLaunchKernelGGL(KernelFunc f, dim3 gridDim, dim3 blockDim, 
                                    unsigned int sharedMemBytes, hipStream_t stream, 
                                    Args... args) {
    // Mock kernel launch - just return success
    return hipSuccess;
}