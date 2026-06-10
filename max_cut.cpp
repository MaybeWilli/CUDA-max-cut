#include "max_cut.h"

constexpr int mode = 0;

MaxCut::MaxCut(int nodes) : nodes(nodes)
{
    config = new int[nodes];
    for (int i = 0; i < nodes; i++)
    {
        config[i] = rand() % 2;
    }
    weight = 0;
    create_graph(mode);

    weight = get_weight();

    gains = new int[nodes];
    for (int i = 0; i < nodes; i++)
    {
        gains[i] = calculate_gain(i);
    }

    cout<<"Starting weight: "<<weight<<endl;

    max_config = new int[nodes];
    max_weight = 0;
}

MaxCut::MaxCut(int nodes, int* edges, int* offsets, int* weights) : nodes(nodes)
{
    this->edges = new int[offsets[nodes]];
    this->offsets = new int[nodes+1];
    this->weights = new int[offsets[nodes]];
    for (int i = 0; i < offsets[nodes]; i++)
    {
        this->edges[i] = edges[i];
        this->weights[i] = weights[i];
    }

    for (int i = 0; i <= nodes; i++)
    {
        this->offsets[i] = offsets[i];
    }
    config = new int[nodes];
    for (int i = 0; i < nodes; i++)
    {
        config[i] = rand() % 2;
    }
    weight = 0;
    weight = get_weight();

    gains = new int[nodes];
    for (int i = 0; i < nodes; i++)
    {
        gains[i] = calculate_gain(i);
    }

    cout<<"Starting weight: "<<weight<<endl;

    max_config = new int[nodes];
    max_weight = 0;
}

MaxCut::MaxCut(int nodes, int* edges, int* offsets, int* weights, int* config) : nodes(nodes)
{
    this->edges = new int[offsets[nodes]];
    this->offsets = new int[nodes+1];
    this->weights = new int[offsets[nodes]];
    for (int i = 0; i < offsets[nodes]; i++)
    {
        this->edges[i] = edges[i];
        this->weights[i] = weights[i];
    }

    for (int i = 0; i <= nodes; i++)
    {
        this->offsets[i] = offsets[i];
    }
    this->config = new int[nodes];
    for (int i = 0; i < nodes; i++)
    {
        this->config[i] = ((config[int(i/32)] >> (i%32)) & 1);
    }
}

int MaxCut::get_weight()
{
    int weight = 0;
    for (int i = 0; i < nodes; i++)
    {
        for (int j = offsets[i]; j < offsets[i+1]; j++)
        {
            if ((config[edges[j]] ^ config[i]) == 1)
            {
                weight += weights[j];
            }
        }
    }
    return weight/2;
}

void MaxCut::create_graph(int mode)
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
            //deg[i] = 1;
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
   create_undirected_graph(temp_offsets, temp_edges, temp_weights);
   delete[] temp_offsets;
   delete[] temp_edges;
   delete[] temp_weights;
}

void MaxCut::create_undirected_graph(int* temp_offsets, int* temp_edges, int* temp_weights)
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

bool MaxCut::solve()
{
    int max = 0;
    int max_index = -1;
    for (int i = 0; i < nodes; i++)
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
    config[index] ^= 1;
    for (int i = offsets[index]; i < offsets[index+1]; i++)
    {
        int v = edges[i];
        if (config[index] == config[v])
        {
            gains[v] += 2 * weights[i];
        }
        else
        {
            gains[v] -= 2 * weights[i];
        }
    }
    gains[index] = -gains[index];
}

int MaxCut::calculate_gain(int candidate)
{
    int c = config[candidate];
    int gain = 0;
    for (int i = offsets[candidate]; i < offsets[candidate+1]; i++)
    {
        if (c == config[edges[i]])
        {
            gain += weights[i];
        }
        else
        {
            gain -= weights[i];
        }
    }

    return gain;
}

void MaxCut::save_max_config()
{
    for (int i = 0; i < nodes; i++)
    {
        max_config[i] = config[i];
    }
    max_weight = weight;
}

int MaxCut::get_max_weight()
{
    int weight = 0;
    for (int i = 0; i < nodes; i++)
    {
        for (int j = offsets[i]; j < offsets[i+1]; j++)
        {
            if ((max_config[edges[j]] ^ max_config[i]) == 1)
            {
                weight += weights[j];
            }
        }
    }
    return weight/2;
}

int MaxCut::perfect_solve()
{
    for (int i = 0; i < (1 << nodes); i++)
    {
        for (int j = 0; j < nodes; j++)
        {
            bool mask = (i >> j) & 1;
            config[j] = mask;
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
    MaxCut maxCut2 = MaxCut(5000);
    MaxCut maxCut = MaxCut(maxCut2.nodes, maxCut2.edges, maxCut2.offsets, maxCut2.weights);
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
                maxCut.flip_vertex(rand() % maxCut.nodes);
            }
        }
        iter -= 1;
    }
    maxCut2.perfect_solve();

    cout<<"Max cut: "<<maxCut.weight<<endl;
    cout<<"Max cut: "<<maxCut.get_weight()<<endl;
    int edge_total = 0;
    for (int i = 0; i < maxCut.offsets[maxCut.nodes]; i++)
    {
        //cout<<maxCut.weights[i]<<" ";
        edge_total += maxCut.weights[i];
        
    }
    cout<<endl;
    cout<<"Edge total: "<<edge_total/2<<endl;

    cout<<"Max cut found: "<<maxCut.max_weight<<endl;
    cout<<"Max cut found: "<<maxCut.get_max_weight()<<endl;
    cout<<"Perfect solve max cut: "<<maxCut2.max_weight<<endl;
    cout<<"Perfect solve max cut: "<<maxCut2.get_max_weight()<<endl;

}//*/