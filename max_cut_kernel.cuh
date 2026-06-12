#ifndef MAX_CUT_KERNEL_CUH
#define MAX_CUT_KERNEL_CUH

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

constexpr int threads = 256;
constexpr int blocks = 32;

inline
cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void max_cut_init(int* edges, int* offsets, int* config, int* current_weight, int* weights, 
    int nodes, int config_size);

__global__ void max_cut_solve(int* edges, int* offsets, int* config, int* current_weight, int* weights,
    int nodes, int config_size);


#endif