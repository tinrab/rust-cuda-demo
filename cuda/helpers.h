#include <cuda_runtime.h>

#define CHECK_CUDA(error)           \
    {                               \
        if (error != cudaSuccess) { \
            return error;           \
        }                           \
    }
