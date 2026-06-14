#include "max_cut_kernel.cuh"

__global__ void max_cut_init(int* edges, int* offsets, int* config, int* current_weight, int* weights, 
    int nodes, int config_size)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int block = blockIdx.x;
    int idx = threadIdx.x;
    int offset = blockIdx.x*(config_size);

    int local_weight = 0;

    for (int v = idx; v < nodes; v += blockDim.x)
    {
        for (int j = offsets[v]; j < offsets[v+1]; j++)
        {
            int u = edges[j];
            if (((config[offset+int(v/32)] >> (v%32)) & 1) ^ ((config[offset+int(u/32)] >> (u%32)) & 1))
            {
                local_weight += weights[j];
            }
        }
    }

    __syncthreads();
    __shared__ int local_weights[threads/32];

    for (int delta = 16; delta > 0; delta >>= 1)
    {
        local_weight += __shfl_down_sync(0xffffffff, local_weight, delta);
    }
    if (lane == 0)
    {
        local_weights[warp] = local_weight;
    }

    __syncthreads();
    for (int stride = blockDim.x/32 / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            local_weights[threadIdx.x] += local_weights[threadIdx.x + stride];
        }

        __syncthreads();
    }

    //write to global
    if(threadIdx.x == 0)
    {
        current_weight[block] = local_weights[0]/2;
    }
}

__global__ void max_cut_solve(int* edges, int* offsets, int* config, int* current_weight, int* weights,
    int nodes, int config_size, int iterations)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int block = blockIdx.x;
    int offset = blockIdx.x*(config_size);

    for (int iteration = 0; iteration < iterations; iteration++)
    {
        unsigned int rng = blockIdx.x * 1234567u +
                   warp * 89123u +
                   iteration;

        int v;
        if (lane == 0)
        {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng ^= threadIdx.x * 0x9e3779b9u;

            v = rng % nodes;
        }

        v = __shfl_sync(0xffffffff, v, 0);
        int c = ((config[offset+int(v/32)] >> (v%32)) & 1);
        int local_gain = 0;
        for (int i = offsets[v]+lane; i < offsets[v+1]; i += 32)
        {
            int u = edges[i];
            if (c == ((config[offset+int(u/32)] >> (u%32)) & 1))
            {
                local_gain += weights[i];
            }
            else
            {
                local_gain -= weights[i];
            }
        }

        for (int delta = 16; delta > 0; delta >>= 1)
        {
            local_gain += __shfl_down_sync(0xffffffff, local_gain, delta);
        }
        if (lane == 0)
        {
            if (local_gain > 0)
            {
                int word_idx = offset + (v >> 5);
                int mask = 1 << (v & 31);

                int old_word = config[word_idx];

                if (atomicCAS(&config[word_idx],
                            old_word,
                            old_word ^ mask) == old_word)
                {
                    atomicAdd(current_weight + block, local_gain);
                }
            }
            else
            {
                float r = (rng & 0x00FFFFFF) * (1.0f / 16777216.0f);
                float log_r = __logf(r);
                float T0 = nodes/100;
                float T = T0 * powf(0.99999f, iteration);

                if (local_gain > T * log_r)
                {
                    int word_idx = offset + (v >> 5);
                    int mask = 1 << (v & 31);

                    int old_word = config[word_idx];

                    if (atomicCAS(&config[word_idx],
                                old_word,
                                old_word ^ mask) == old_word)
                    {
                        atomicAdd(current_weight + block, local_gain);
                    }
                }
            }
        }
    }
}