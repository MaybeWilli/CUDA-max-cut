#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

/*
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
int* current_weight;*/
struct Graph
{
    int nodes;
    vector<int> edges;
    vector<int> offsets;
    vector<int> weights;
};

void create_graph(Graph& graph, int nodes, int mode);