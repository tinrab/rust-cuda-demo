#ifndef LIB_MATH_H
#define LIB_MATH_H

#include <cuda_runtime.h>

extern "C" {
    cudaError_t math_vector_add(
        int n, const float* a, const float* b, float* c
    );
}

#endif  // LIB_MATH_H
