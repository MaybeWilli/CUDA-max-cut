#include <stdio.h>
#include <assert.h>
#include <array>
#include <iostream>
#include <vector>
#include "max_cut.h"
#include <chrono>
#include <algorithm>

//constexpr int nodes = 50000;
constexpr int nodes = 125*125;
constexpr int config_size = int(nodes/32)+1;
constexpr int blocks = 32;
int mode = 1;
int* vertices;
int* edges;
int* offsets;
int* config;
int* weights;
int weight;
int* current_weight;

constexpr int threads = 256;

using namespace std;

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

void create_undirected_graph(int* temp_offsets, int* temp_edges, int* temp_weights)
{
    int* deg = new int[nodes+1]();
    for (int v = 0; v < nodes; v++)
    {
        for (int j = temp_offsets[v]; j < temp_offsets[v+1]; j++)
        {
            int u = temp_edges[j];
            deg[u]++;
            deg[v]++;
        }
    }

    int total = 0;
    offsets = new int[nodes+1]();
    int* indexes = new int[nodes+1]();
    for (int i = 0; i < nodes; i++)
    {
        offsets[i] = total;
        total += deg[i];
        indexes[i] = offsets[i];
    }

    int total_edges = temp_offsets[nodes]*2;
    offsets[nodes] = total_edges;
    edges = new int[total_edges];
    weights = new int[total_edges];
    

    for (int v = 0; v < nodes; v++)
    {
        for (int j = temp_offsets[v]; j < temp_offsets[v+1]; j++)
        {
            int u = temp_edges[j];
            int weight = temp_weights[j];
            edges[indexes[u]] = v;
            edges[indexes[v]] = u;
            weights[indexes[u]] = weight;
            weights[indexes[v]] = weight;
            indexes[u]++;
            indexes[v]++;
        }
    }
    delete[] deg;
    delete[] indexes;
}

void create_graph()
{
    int* temp_edges;
    int* temp_offsets;
    int* temp_weights;
    if (mode == 0)
    {
        int* deg = new int[nodes];
        int edge_total = 0;
        for (int i = 0; i < nodes; i++)
        {
            deg[i] = rand() % 4 + 4;
            edge_total += deg[i];
        }

        temp_edges = new int[edge_total];
        temp_weights = new int[edge_total];
        temp_offsets = new int[nodes+1];
        int offset = 0;
        for (int i = 0; i < nodes; i++)
        {
            temp_offsets[i] = offset;
            for (int j = 0; j < deg[i]; j++)
            {
                int v = rand() % nodes;
                if (v == i)
                {
                    v = (v + 2) % nodes;
                }
                temp_edges[offset + j] = v;
                temp_weights[offset + j] = rand() % 10 + 1;
            }
            offset += deg[i];

        }
        temp_offsets[nodes] = edge_total;

    }
    else if (mode == 1)
    {
        int l = sqrt(nodes);
        temp_edges = new int[nodes*2];
        temp_offsets = new int[nodes+1];
        temp_weights = new int[nodes*2];

        for (int i = 0; i < nodes*2; i += 2)
        {
            temp_offsets[i/2] = i;
            temp_edges[i] = (i/4 % l + 1) % l + (int(i/4/l)) * l;
            //edges[i+1] = (i/4 % l - 1 + l) % l + (int(i/4/l)) * l;
            temp_edges[i+1] = (i/4 % l) + ((int(i/4/l)+1) % l) * l;
            //edges[i+3] = (i/4 % l) + ((int(i/4/l)-1+l) % l) * l;
            temp_weights[i] = rand() % 10 + 1;
            temp_weights[i+1] = rand() % 10 + 1;
        }

        temp_offsets[nodes] = nodes*2;
    }
    else if (mode == 2)
    {
        int* deg = new int[nodes];

        int stub_count = 0;
        double gamma = 2.5;

        for (int i = 0; i < nodes; i++)
        {
            double u = (rand() + 1.0) / (RAND_MAX + 1.0);

            deg[i] = std::max(
                1,
                (int)pow(u, -1.0 / (gamma - 1))
            );

            deg[i] = std::min(
                deg[i],
                int(sqrt(nodes))
            );

            stub_count += deg[i];
        }

        if (stub_count & 1)
        {
            deg[rand() % nodes]++;
            stub_count++;
        }

        std::vector<int> stubs;
        stubs.reserve(stub_count);

        for (int v = 0; v < nodes; v++)
        {
            for (int j = 0; j < deg[v]; j++)
            {
                stubs.push_back(v);
            }
        }

        std::mt19937 rng(rand());
        std::shuffle(stubs.begin(), stubs.end(), rng);

        int edge_total = stub_count / 2;

        temp_edges = new int[edge_total];
        temp_weights = new int[edge_total];
        temp_offsets = new int[nodes + 1]();

        for (int i = 0; i < stub_count; i += 2)
        {
            int u = stubs[i];
            temp_offsets[u + 1]++;
        }

        for (int i = 1; i <= nodes; i++)
        {
            temp_offsets[i] += temp_offsets[i - 1];
        }

        std::vector<int> ptr(nodes);

        for (int i = 0; i < nodes; i++)
        {
            ptr[i] = temp_offsets[i];
        }

        for (int i = 0; i < stub_count; i += 2)
        {
            int u = stubs[i];
            int v = stubs[i + 1];

            if (u == v)
            {
                v = (v + 1) % nodes;
            }

            int pos = ptr[u]++;

            temp_edges[pos] = v;
            temp_weights[pos] = rand() % 10 + 1;
        }

        delete[] deg;
    }//*/
    create_undirected_graph(temp_offsets, temp_edges, temp_weights);
    delete[] temp_offsets;
    delete[] temp_edges;
    delete[] temp_weights;
}

__global__ void max_cut_init(int* edges, int* offsets, int* config, int* current_weight, int* weights)
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
    for (int stride = threads/32 / 2; stride > 0; stride >>= 1)
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

__global__ void max_cut_solve(int* edges, int* offsets, int* config, int* current_weight, int* weights)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int block = blockIdx.x;
    int idx = threadIdx.x;
    int offset = blockIdx.x*(config_size);

    int iterations = nodes*50;
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

void init_input()
{
    config = new int[config_size * blocks];
    for (int i = 0; i < config_size * blocks; i++)
    {
        config[i] = rand();
    }
    weight = 0;
    create_graph();

    cout<<"Starting weight: "<<weight<<endl;

    current_weight = new int[blocks];
}

int main()
{
    unsigned int seed = time(NULL);
    srand(seed);

    init_input();
    int* d_edges;
    int* d_offsets;
    int* d_config;
    int* d_current_weight;
    int* d_weights;
    MaxCut maxCut = MaxCut(nodes, edges, offsets, weights);

    checkCuda ( cudaMalloc((void**)&d_edges, offsets[nodes] * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_offsets, (nodes+1) * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_config, config_size * blocks * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_current_weight, blocks * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_weights, offsets[nodes] * sizeof(int)));

    checkCuda( cudaMemcpy(d_edges, edges, offsets[nodes] * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_offsets, offsets, (nodes+1) * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_config, config, config_size * blocks * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_weights, weights, offsets[nodes] * sizeof(int), cudaMemcpyHostToDevice) );

    dim3 grid(blocks, 1);
    dim3 block(threads, 1);

    float milliseconds;
    cudaEvent_t startEvent, stopEvent;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );

    checkCuda( cudaEventRecord(startEvent, 0) );
    max_cut_init<<<grid, block>>>(d_edges, d_offsets, d_config, d_current_weight, d_weights);
    max_cut_solve<<<grid, block>>>(d_edges, d_offsets, d_config, d_current_weight, d_weights);
    max_cut_init<<<grid, block>>>(d_edges, d_offsets, d_config, d_current_weight, d_weights);
    
    checkCuda(cudaDeviceSynchronize());
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );

    checkCuda( cudaMemcpy(config, d_config, config_size * blocks * sizeof(int), cudaMemcpyDeviceToHost) );
    checkCuda( cudaMemcpy(current_weight, d_current_weight, blocks * sizeof(int), cudaMemcpyDeviceToHost) );

    int max_found_weight = 0;
    int max_index = 0;
    cout<<"Milliseconds: "<<milliseconds<<endl;
    for (int i = 0; i < blocks; i++)
    {
        if (current_weight[i] > max_found_weight)
        {
            max_found_weight = current_weight[i];
            max_index = i;
        }
        //cout<<"Starting weight: "<<max_weight[i]<<" current weight: "<<current_weight[i]<<endl;
    }
    cout<<"Max gpu weight: "<<max_found_weight<<endl;
    cout<<"Verifying answer on cpu"<<endl;
    MaxCut check = MaxCut(nodes, edges, offsets, weights, config+max_index*config_size);
    int expected = check.get_weight();
    cout<<"Expected: "<<expected<<" got "<<max_found_weight<<endl;
    if (expected == max_found_weight)
    {
        cout<<"Expected result matches"<<endl;
    }
    else
    {
        cout<<"Error found"<<endl;
    }

    int iter = nodes*50;

    auto start_time = chrono::steady_clock::now();
    while (iter > 0)
    {
        bool needs_flip = maxCut.solve();

        //cout<<maxCut.weight<<" "<<maxCut.get_weight()<<endl;
        if (needs_flip)
        {
            if (maxCut.weight > maxCut.max_weight)
            {
                maxCut.save_max_config();
            }
            int flips = 25;
            for (int i = 0; i < flips; i++)
            {
                maxCut.flip_vertex(rand() % maxCut.nodes);
            }
        }
        iter -= 1;
    }
    auto end_time = chrono::steady_clock::now();
    chrono::duration<double> time_passed = end_time - start_time;

    double max_cpu_weight = max(maxCut.get_max_weight(), maxCut.get_weight());
    cout<<"Max cpu cut found: "<<max_cpu_weight<<endl;
    std::cout<<"CPU milliseconds: "<<time_passed.count()*1000<<std::endl;
    cout<<"GPU vs cpu performance: "<<max_found_weight*1.0/max_cpu_weight<<endl;//*/


}