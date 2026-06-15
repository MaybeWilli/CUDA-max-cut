# CUDA-max-cut

CUDA kernel for GPU-accelerated max-cut. Max-cut is an NP-hard graph optimization problem where vertices in a graph are partitioned into two groups to maximize the weight of the edges crossing between the partition. GPU kernel uses warp-parallel, frontier-less stochastic local search for Max-Cut using simulated annealing–style acceptance with atomic graph state updates. 

Each iteration samples a random vertex and evaluates the cut gain of flipping its partition assignment. The neighborhood evaluation is performed cooperatively at the warp level using shuffle-based reductions over adjacency lists.

Moves that improve the cut are accepted greedily, while non-improving moves may be accepted probabilistically based on a temperature schedule, allowing the algorithm to escape local minima.

## System Overview

The graph is represented in CSR format. Config is represented in bit-packed 32-bit integers. Each block is divided into 8 warps, each of which randomly chooses a bit to flip, and accepts positive gain flips greedily. Negative gains are accepted at a probability controlled by a temperature parameter. Gain calculation is accelerated using the shufle_sync warp primitive. 32 total blocks are launched per kernel, allowing for more search trajectories.

## Build Instructions

### Option 1: 
Using build script:

```
cd CUDA-max-cut
sudo chmod +x ./build.sh
./build.sh
```

### Option 2:

Using CMakeList:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Max Cut Performance (GPU vs CPU)

All entries are averaged over 10 runs. Both CPU and GPU use 50xnodes iterations by default.
Speedup = CPU time / GPU time  
Improvement = (GPU cut weight − CPU cut weight) / CPU cut weight × 100%

| Graph Type | Nodes  | GPU Weight | CPU Weight | GPU (ms) | CPU (ms) | Speedup | Improvement |
|------------|--------|------------|------------|----------|----------|---------|-------------|
| Power-law  | 10000  | 61430.6    | 60579.4    | 455.94   | 2280.45  | 5.01x   | 1.57%       |
| Power-law  | 20000  | 122052.3   | 119961.7   | 898.68   | 9092.06  | 10.12x  | 1.77%       |
| Power-law  | 50000  | 309653.1   | 305588.9   | 2231.18  | 56800.27 | 25.66x  | 1.33%       |

| Graph Type | Nodes | GPU Weight | CPU Weight | GPU (ms) | CPU (ms) | Speedup | Improvement |
|------------|-------|------------|------------|----------|----------|---------|-------------|
| Grid       | 10201 | 109025.5   | 103303.1   | 463.40   | 2376.66  | 5.13x   | 5.84%       |
| Grid       | 20449 | 216326.0   | 204980.9   | 911.78   | 9495.39  | 10.41x  | 5.40%       |
| Grid       | 50625 | 524272.6   | 506605.6   | 2219.50  | 57983.66 | 26.13x  | 3.60%       |

| Graph Type | Nodes | GPU Weight | CPU Weight | GPU (ms) | CPU (ms) | Speedup | Improvement |
|------------|-------|------------|------------|----------|----------|---------|-------------|
| Sparse     | 10000 | 225442.8   | 222591.7   | 437.89   | 2297.71  | 5.25x   | 1.28%       |
| Sparse     | 20000 | 449256.1   | 443858.8   | 891.36   | 9204.11  | 10.32x  | 1.22%       |
| Sparse     | 50000 | 1113419.7  | 1107076.4  | 2821.71  | 56678.59 | 20.09x  | 0.60%       |

## Running Instructions

The executable produced is CUDA-max-cut/max_cut. It comes with four command-line arguments: 
  nodes: determines the amount of nodes in the graph. 
  mode: can be sparse, grid, or powerlaw, depending on the desired graph type
  iterations: number of iterations. Defaults to nodes*50
  compare: can be cpu, gpu, compare, or verify. CPU runs CPU only. GPU runs GPU only. Verify runs GPU, then verifies the answer with CPU, and Compare runs both and compares peformances.
  help: shows all arguments

Example:

```
./max_cut --nodes 10000 --graph powerlaw
```
  

## Notes

### Test Script

To run test script:

```
sudo chmod +x ./test_script.sh
./test_script.sh
```

## CPU Baseline

CPU baseline uses a single-threaded gain-cached algorithm that uses greedy hill-climbing until it reaches a local peak. Pertubations are used for greater search space.

### Hardware Used

GPU: GeForce RTX 3060
CPU: AMD Ryzen 5 5600X
