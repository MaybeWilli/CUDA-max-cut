#include <stdio.h>
#include <assert.h>
#include <array>
#include <iostream>
#include <vector>
#include "max_cut.h"
#include <chrono>
#include <algorithm>
#include "max_cut_kernel.cuh"
#include <string>

using namespace std;

int main(int argc, char** argv)
{
    unsigned int seed = time(NULL);
    srand(seed);
    bool default_iterations = true;
    int nodes = 125*125;
    int mode = 1;
    int iterations;


    string compare = "compare";

    for (int i = 1; i < argc; i+=2)
    {
        if (string(argv[i]) == "--graph")
        {
            string graph_type = string(argv[i+1]);
            if (graph_type == "sparse")
            {
                mode = 0;
            }
            else if (graph_type == "grid")
            {
                mode = 1;
            }
            else
            {
                mode = 2;
            }
        }
        else if (string(argv[i]) == "--mode")
        {
            compare = string(argv[i+1]);
        }
        else if (string(argv[i]) == "--nodes")
        {
            nodes = stoi(argv[i+1]);
        }
        else if (string(argv[i]) == "--iterations")
        {
            iterations = stoi(argv[i+1]);
            default_iterations = false;
        }
        else if (string(argv[i]) == "--help")
        {
            printf("Flags: \n");
            printf("    --graph:         Values: sparse, grid, or powerlaw. Determines grid type.\n");
            printf("    --mode:          Values: cpu, gpu, compare, verify. Determines if algorithm is"
                   "\n                   run on cpu, gpu, if it's compared, and if gpu answer is verified on cpu.\n");
            printf("    --nodes:         Number of vertices in the graph.\n");
            printf("    --iterations:    Number of max-cut iterations. Defaults to nodes*50.\n");
            printf("\n\nExample usage: ./max_cut --nodes 5000 --graph powerlaw --mode compare --iterations 250000");
        }

    }

    if (default_iterations)
    {
        iterations = nodes*50;
    }

    int config_size = int(nodes/32)+1;

    Graph graph;
    create_graph(graph, nodes, mode);

    vector<int> current_config(blocks*config_size);
    for (int i = 0; i < blocks*config_size; i++)
    {
        current_config[i] == rand();
    }
    vector<int> current_weight(blocks);
    vector<int> max_weights = current_config;
    vector<int> max_configs(blocks*config_size);
    int* d_edges;
    int* d_offsets;
    int* d_config;
    int* d_current_weight;
    int* d_weights;
    MaxCut maxCut = MaxCut(graph);
    bool verify;
    int expected;
    int max_found_weight = 0;
    int max_index = 0;
    float milliseconds;

    vector<string> graph_types = {"sparse", "grid", "power-law"};

    printf("Graph: Nodes: %i, graph type: %s, iterations: %d\n", nodes, graph_types[mode].c_str(), iterations);
    if (compare != "cpu")
    {
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

        cudaEvent_t startEvent, stopEvent;
        checkCuda( cudaEventCreate(&startEvent) );
        checkCuda( cudaEventCreate(&stopEvent) );

        checkCuda( cudaEventRecord(startEvent, 0) );
        max_cut_init<<<grid, block>>>(d_edges, d_offsets, d_config, d_current_weight, d_weights, nodes, config_size);
        max_cut_solve<<<grid, block>>>(d_edges, d_offsets, d_config, d_current_weight, d_weights, nodes, config_size, iterations);
        max_cut_init<<<grid, block>>>(d_edges, d_offsets, d_config, d_current_weight, d_weights, nodes, config_size);
        
        checkCuda(cudaDeviceSynchronize());
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );

        checkCuda( cudaMemcpy(current_config.data(), d_config, config_size * blocks * sizeof(int), cudaMemcpyDeviceToHost) );
        checkCuda( cudaMemcpy(current_weight.data(), d_current_weight, blocks * sizeof(int), cudaMemcpyDeviceToHost) );

        checkCuda ( cudaFree(d_edges));
        checkCuda ( cudaFree(d_offsets));
        checkCuda ( cudaFree(d_config));
        checkCuda ( cudaFree(d_current_weight));
        checkCuda ( cudaFree(d_weights));

        for (int i = 0; i < blocks; i++)
        {
            if (current_weight[i] > max_found_weight)
            {
                max_found_weight = current_weight[i];
                max_index = i;
            }
        }
        printf("GPU    Weight = %d    Time = %.2f (ms)\n", max_found_weight, milliseconds);

        if (compare != "gpu")
        {
            vector<int> biggest_config(config_size);
            for (int i = 0; i < config_size; i++)
            {
                biggest_config[i]= current_config[max_index*config_size+i];
            }
            MaxCut check = MaxCut(graph, biggest_config);
            expected = check.get_weight();
            verify = expected == max_found_weight;
        }
    }

    double cpu_time = 0;
    int cpu_weight = 0;
    if (compare == "compare" || compare == "cpu")
    {
        int iter = iterations;

        auto start_time = chrono::steady_clock::now();
        while (iter > 0)
        {
            bool needs_flip = maxCut.solve();

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

        cpu_weight = max(maxCut.get_max_weight(), maxCut.get_weight());
        cpu_time = time_passed.count()*1000;
        printf("CPU    Weight = %d    Time = %.2f (ms)\n", cpu_weight, cpu_time);

        if (compare == "compare")
        {
            printf("\nGPU speedup over cpu: %.2fx\nGPU solution improvement over cpu: %.2f%%\n", 
                cpu_time*1.0/milliseconds, (max_found_weight*1.0/cpu_weight)*100-100);
        }
    }

    if (compare == "compare" || compare == "verify")
    {
        if (verify)
        {
            printf("Verification: PASS\n");
        }
        else
        {
            printf("Verification failed. Expected %d, got %d\n", expected, max_found_weight);
        }
    }


}