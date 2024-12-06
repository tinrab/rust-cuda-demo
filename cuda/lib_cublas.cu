#include "helpers.h"
#include "lib_cublas.h"

cudaError_t cublas_sgemm(
    int n, const float* a, const float* b, float* c, float alpha, float beta,
    cublasStatus_t* cublas_status
) {
    int n2 = n * n;
    // Device memory
    float* device_a = 0;
    float* device_b = 0;
    float* device_c = 0;

    cublasHandle_t handle;
    *cublas_status = cublasCreate(&handle);
    if (*cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_a), n2 * sizeof(device_a[0])
    ));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_b), n2 * sizeof(device_b[0])
    ));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_c), n2 * sizeof(device_c[0])
    ));

    // Copy data from host to device
    *cublas_status = cublasSetVector(n2, sizeof(a[0]), a, 1, device_a, 1);
    if (*cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }
    *cublas_status = cublasSetVector(n2, sizeof(b[0]), b, 1, device_b, 1);
    if (*cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }
    *cublas_status = cublasSetVector(n2, sizeof(c[0]), c, 1, device_c, 1);
    if (*cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    // Perform matrix multiplication
    *cublas_status = cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, device_a, n,
        device_b, n, &beta, device_c, n
    );
    if (*cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    // Copy data from device to host
    *cublas_status = cublasGetVector(n2, sizeof(c[0]), device_c, 1, c, 1);
    if (*cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    // Free resources
    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    *cublas_status = cublasDestroy(handle);
    if (*cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}
