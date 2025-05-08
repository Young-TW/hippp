#include <hip/hip_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int vector_add_origin() {
    const int N = 1024;
    const int size = N * sizeof(float);

    float *a, *b, *c;
    hipMalloc((void**)&a, size);
    hipMalloc((void**)&b, size);
    hipMalloc((void**)&c, size);

    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
    }

    hipMemcpy(a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(b, h_b, size, hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vector_add<<<numBlocks, blockSize>>>(a, b, c, N);

    hipMemcpy(h_c, c, size, hipMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %f + %f != %f\n", i, h_a[i], h_b[i], h_c[i]);
            break;
        }
    }

    hipFree(a);
    hipFree(b);
    hipFree(c);

    return 0;
}