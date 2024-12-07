#include "helpers.h"
#include "lib_math.h"

__global__ void vector_add_kernel(
    int n,
    const float* a,
    const float* b,
    float* c
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

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);

    vector_add_kernel<<<grid, block>>>(n, device_a, device_b, device_c);

    // void* args[] = {
    //     reinterpret_cast<void*>(&n),
    //     reinterpret_cast<void*>(&device_a),
    //     reinterpret_cast<void*>(&device_b),
    //     reinterpret_cast<void*>(&device_c),
    // };
    // cudaLaunchKernel((void*)vector_add_kernel, grid, block, args);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(c, device_c, BYTES, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    return cudaSuccess;
}
