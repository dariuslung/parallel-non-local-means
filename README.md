# Parallelization of Non-Local Means
This project implements the Non-Local Means (NLM) algorithm for the course of NYCU, Parallel Programming - Fall '25. It provides a performance comparison between a standard CPU implementation, a multi-threaded CPU implementation (OpenMP), and two GPU implementations using NVIDIA CUDA (Global Memory vs. Shared Memory).

## Project Overview
Non-Local Means is a powerful denoising technique that preserves textures by averaging similar patches from across the image. However, its high computational cost $O(N^4)$ makes it slow on standard CPUs. This project accelerates the algorithm using:
1. OpenMP: Multi-threading for CPU.
2. CUDA (Global Memory): Massively parallel GPU computing.
3. CUDA (Shared Memory): Optimizing GPU memory bandwidth by caching image patches.

## Requirements
- Compiler: g++ (with OpenMP support)
- CUDA Toolkit: nvcc (NVIDIA CUDA Compiler)
- Make: Build system

## Compilation
To build the project, simply run make in the root directory.
```
make
```

## Usage
The program is run from the command line. You can select the execution mode and tune the NLM parameters using the flags below.

### Syntax
```
./build/main [options]
```

### Options
| Flag | Long Flag      | Description                            | Default |
| :----| :------------: | :------------------------------------: | ------: |
| -m   | --mode         | Execution Mode <0-3>                   | 0       | 
| -n   | --image-num    | ID of the input image to use <0-3>     | 0       | 
| -p   | --patch-size   | Size of the square patch (k×k)         | 5       | 
| -f   | --filter-sigma | Smoothing parameter h (controls decay) | 0.06    | 
| -s   | --patch-sigma  | Standard deviation for patch Gaussian  | 0.8     | 
| -h   | --help         | Show usage information                 | N/A     | 

### Execution Modes (`-m`)
- `0` **CPU Serial**: Baseline single-threaded C++ implementation. Slowest, used for correctness verification.
- `1` **CPU Parallel**: Multi-threaded implementation using OpenMP. Parallelizes the outer loops.
- `2` **GPU Global Memory**: Naive CUDA implementation. Each thread computes one pixel, reading patches directly from Global Memory.
- `3` **GPU Shared Memory**: Optimized CUDA implementation. Threads cooperatively load search windows into Shared Memory (L1 Cache) to minimize global memory bandwidth bottlenecks.

### Input Image (`-n`)
- `0` **House**: 64x64
- `1` **Flower**: 128x128
- `2` **Lena_256**: 256x256
- `3` **Lena_512**: 512x512
