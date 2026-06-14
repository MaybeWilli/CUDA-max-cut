cmake_minimum_required(VERSION 3.18)

project(max_cut LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(max_cut
    src/max_cut.cu
    src/max_cut.cpp
    src/graph.cpp
    src/max_cut_kernel.cu
)

target_compile_options(max_cut PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -Xcompiler=-O3,-march=native
    >
)