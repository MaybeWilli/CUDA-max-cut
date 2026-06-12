#include <stdio.h>
#include <assert.h>
#include <array>
#include <iostream>
#include <vector>
#include "max_cut.h"
#include <chrono>
#include <algorithm>
#include "max_cut_kernel.cuh"

//constexpr int nodes = 50000;
/*constexpr int nodes = 125*125;
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

constexpr int threads = 256;*/

using namespace std;

int main()
{
    unsigned int seed = time(NULL);
    srand(seed);
    int nodes = 125*125;
    int mode = 1;

    int config_size = int(nodes/32)+1;

    Graph graph;
    create_graph(graph, nodes, mode);

    vector<int> current_config(blocks*config_size);
    vector<int> current_weight(blocks);
    int* d_edges;
    int* d_offsets;
    int* d_config;
    int* d_current_weight;
    int* d_weights;
    MaxCut maxCut = MaxCut(graph);

    checkCuda ( cudaMalloc((void**)&d_edges, graph.offsets[nodes] * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_offsets, (nodes+1) * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_config, config_size * blocks * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_current_weight, blocks * sizeof(int)));
    checkCuda ( cudaMalloc((void**)&d_weights, graph.offsets[nodes] * sizeof(int)));

    checkCuda( cudaMemcpy(d_edges, graph.edges.data(), graph.offsets[nodes] * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_offsets, graph.offsets.data(), (nodes+1) * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_config, current_config.data(), config_size * blocks * sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_weights, graph.weights.data(), graph.offsets[nodes] * sizeof(int), cudaMemcpyHostToDevice) );

    dim3 grid(blocks, 1);
    dim3 block(threads, 1);

    float milliseconds;
    cudaEvent_t startEvent, stopEvent;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );

    checkCuda( cudaEventRecord(startEvent, 0) );
    max_cut_init<<<grid, block>>>(d_edges, d_offsets, d_config, d_current_weight, d_weights, nodes, config_size);
    max_cut_solve<<<grid, block>>>(d_edges, d_offsets, d_config, d_current_weight, d_weights, nodes, config_size);
    max_cut_init<<<grid, block>>>(d_edges, d_offsets, d_config, d_current_weight, d_weights, nodes, config_size);
    
    checkCuda(cudaDeviceSynchronize());
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );

    checkCuda( cudaMemcpy(current_config.data(), d_config, config_size * blocks * sizeof(int), cudaMemcpyDeviceToHost) );
    checkCuda( cudaMemcpy(current_weight.data(), d_current_weight, blocks * sizeof(int), cudaMemcpyDeviceToHost) );

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
    vector<int> biggest_config(config_size);
    for (int i = 0; i < config_size; i++)
    {
        biggest_config[i]= current_config[max_index*config_size+i];
    }
    MaxCut check = MaxCut(graph, biggest_config);
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
                maxCut.flip_vertex(rand() % maxCut.graph.nodes);
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