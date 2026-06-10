#ifndef MAX_CUT_H
#define MAX_CUT_H

#include <vector>
#include <random>
#include <iostream>

using namespace std;

class MaxCut
{
    public:
        int* vertices;
        int* edges;
        int* offsets;
        int nodes;
        int* config;
        int* weights;
        int weight;
        int* gains;

        int* max_config;
        int max_weight;

        MaxCut(int nodes);
        MaxCut(int nodes, int* edges, int* offsets, int* weights);
        MaxCut(int nodes, int* edges, int* offsets, int* weights, int* config);
        void create_graph(int mode);
        void create_undirected_graph(int* temp_offsets, int* temp_edges, int* temp_weights);
        bool solve();
        int calculate_gain(int candidate);
        void display();
        void flip_vertex(int index);
        int get_weight();

        void save_max_config();
        int get_max_weight();

        int perfect_solve();
};

#endif