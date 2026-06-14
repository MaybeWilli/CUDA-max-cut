# CUDA-max-cut

CUDA kernel for GPU-accelerated max-cut. Max-cut is an NP-hard graph optimization problem where vertices in a graph are partitioned into two groups to maximize the weight of the edges crossing between the partition. GPU kernel uses warp-parallel, frontier-less stochastic local search for Max-Cut using simulated annealing–style acceptance with atomic graph state updates. 

This implementation uses a warp-parallel stochastic local search algorithm
for the Max-Cut problem, inspired by simulated annealing and Metropolis
acceptance dynamics.

Each iteration samples a random vertex and evaluates the cut gain of flipping
its partition assignment. The neighborhood evaluation is performed cooperatively
at the warp level using shuffle-based reductions over adjacency lists.

Moves that improve the cut are accepted greedily, while non-improving moves
may be accepted probabilistically based on a temperature schedule, allowing
the algorithm to escape local minima.

## Build Instructions

### Option 1: 
Using build script:

```
cd CUDA-max-cut
chmod +x ./build.sh
./build.sh
```

### Option 2:

Using CMakeList:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Max Cut Performance (GPU vs CPU)
