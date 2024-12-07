#ifndef LIB_CUBLAS_H
#define LIB_CUBLAS_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

extern "C" {
    cudaError_t cublas_sgemm(
        int n,
        const float* a,
        const float* b,
        float* c,
        float alpha,
        float beta,
        cublasStatus_t* cublas_status
    );
}

#endif // LIB_CUBLAS_H
