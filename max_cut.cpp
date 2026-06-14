#include "max_cut.h"

constexpr int mode = 0;

MaxCut::MaxCut(Graph& graph) : graph(graph)
{
    config_size = int(graph.nodes/32)+1;
    config.resize(config_size);
    for (int i = 0; i < config_size; i++)
    {
        config[i] = rand();
    }
    weight = get_weight();

    gains.resize(graph.nodes);
    for (int i = 0; i < graph.nodes; i++)
    {
        gains[i] = calculate_gain(i);
    }

    max_config.resize(config_size);
    max_weight = 0;
}

MaxCut::MaxCut(Graph& graph, vector<int>& config) : graph(graph)
{
    config_size = int(graph.nodes/32)+1;
    this->config = config;
    weight = get_weight();

    gains.resize(graph.nodes);
    for (int i = 0; i < graph.nodes; i++)
    {
        gains[i] = calculate_gain(i);
    }

    max_config.resize(config_size);
    max_weight = 0;
}

int MaxCut::get_weight()
{
    int weight = 0;
    for (int i = 0; i < graph.nodes; i++)
    {
        for (int j = graph.offsets[i]; j < graph.offsets[i+1]; j++)
        {
            int u = graph.edges[j];
            int v = i;
            if ((((config[int(v/32)] >> (v%32)) & 1) ^ ((config[int(u/32)] >> (u%32)) & 1)) == 1)
            {
                weight += graph.weights[j];
            }
        }
    }
    return weight/2;
}

bool MaxCut::solve()
{
    int max = 0;
    int max_index = -1;
    for (int i = 0; i < graph.nodes; i++)
    {
        if (gains[i] > 0 && gains[i] > max)
        {
            max = gains[i];
            max_index = i;
        }
    }

    if (max_index == -1)
    {
        return true;
    }
    else
    {
        flip_vertex(max_index);
        return false;
    }
}

void MaxCut::flip_vertex(int index)
{
    weight += gains[index];
    config[index/32] ^= 1 << (index & 31);
    for (int i = graph.offsets[index]; i < graph.offsets[index+1]; i++)
    {
        int v = graph.edges[i];
        if (((config[int(index/32)] >> (index%32)) & 1) == ((config[int(v/32)] >> (v%32)) & 1))
        {
            gains[v] += 2 * graph.weights[i];
        }
        else
        {
            gains[v] -= 2 * graph.weights[i];
        }
    }
    gains[index] = -gains[index];
}

int MaxCut::calculate_gain(int candidate)
{
    int c = ((config[int(candidate/32)] >> (candidate%32)) & 1);
    int gain = 0;
    for (int i = graph.offsets[candidate]; i < graph.offsets[candidate+1]; i++)
    {
        if (c == ((config[int(graph.edges[i]/32)] >> (graph.edges[i]%32)) & 1))
        {
            gain += graph.weights[i];
        }
        else
        {
            gain -= graph.weights[i];
        }
    }

    return gain;
}

void MaxCut::save_max_config()
{
    for (int i = 0; i < config_size; i++)
    {
        max_config[i] = config[i];
    }
    max_weight = weight;
}

int MaxCut::get_max_weight()
{
    int weight = 0;
    for (int i = 0; i < graph.nodes; i++)
    {
        for (int j = graph.offsets[i]; j < graph.offsets[i+1]; j++)
        {
            if ((((max_config[int(graph.edges[j]/32)] >> (graph.edges[j]%32)) & 1) ^ 
                ((max_config[int(i/32)] >> (i%32)) & 1)) == 1)
            {
                weight += graph.weights[j];
            }
        }
    }
    return weight/2;
}

int MaxCut::perfect_solve()
{
    for (int i = 0; i < (1 << graph.nodes); i++)
    {
        std::fill(config.begin(), config.end(), 0);
        for (int j = 0; j < graph.nodes; j++)
        {
            bool mask = (i >> j) & 1;
            if (mask)
                config[j/32] |= (1 << (j & 31));
            else
                config[j/32] &= ~(1 << (j & 31));
        }

        weight = get_weight();
        if (weight > max_weight)
        {

            save_max_config();
        }
    }
    return max_weight;
}

/*int main()
{
    Graph graph;
    create_graph(graph, 16, 0);
    MaxCut maxCut2 = MaxCut(graph);
    MaxCut maxCut = MaxCut(graph);
    cout<<maxCut.weight<<" "<<maxCut.get_weight()<<endl;
    int iter = 150000;

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
    maxCut2.perfect_solve();

    cout<<"Max cut: "<<maxCut.weight<<endl;
    cout<<"Max cut: "<<maxCut.get_weight()<<endl;
    int edge_total = 0;
    for (int i = 0; i < maxCut.graph.offsets[maxCut.graph.nodes]; i++)
    {
        //cout<<maxCut.weights[i]<<" ";
        edge_total += maxCut.graph.weights[i];
        
    }
    cout<<endl;
    cout<<"Edge total: "<<edge_total/2<<endl;

    cout<<"Max cut found: "<<maxCut.max_weight<<endl;
    cout<<"Max cut found: "<<maxCut.get_max_weight()<<endl;
    cout<<"Perfect solve max cut: "<<maxCut2.max_weight<<endl;
    cout<<"Perfect solve max cut: "<<maxCut2.get_max_weight()<<endl;

}//*/