nvcc -O3 -Xcompiler="-O3 -march=native" src/max_cut.cu src/max_cut.cpp src/graph.cpp src/max_cut_kernel.cu -o max_cut
