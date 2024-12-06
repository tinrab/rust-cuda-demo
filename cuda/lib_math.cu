#include "helpers.h"
#include "lib_math.h"

__global__ void vector_add_kernel(
    int n, const float* a, const float* b, float* c
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

cudaError_t math_vector_add(int n, const float* a, const float* b, float* c) {
    const size_t BYTES = n * sizeof(float);

    float* device_a = 0;
    float* device_b = 0;
    float* device_c = 0;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_a), BYTES));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_b), BYTES));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_c), BYTES));

    CHECK_CUDA(cudaMemcpy(device_a, a, BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, b, BYTES, cudaMemcpyHostToDevice));

    vector_add_kernel<<<(n + 255) / 256, 256>>>(
        n, device_a, device_b, device_c
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(c, device_c, BYTES, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    return cudaSuccess;
}
