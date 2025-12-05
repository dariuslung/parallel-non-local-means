# Parallelization of Non-Local Means
This project implements the Non-Local Means (NLM) algorithm for the course of NYCU, Parallel Programming - Fall '25. It provides a performance comparison between a standard CPU implementation, a multi-threaded CPU implementation (OpenMP), and two GPU implementations using NVIDIA CUDA (Global Memory vs. Shared Memory).


## Project Overview
Non-Local Means is a powerful denoising technique that preserves textures by averaging similar patches from across the image. However, its high computational cost $O(N^4)$ makes it slow on standard CPUs. This project accelerates the algorithm using:
1. **OpenMP:** Multi-threading for CPU.
2. **CUDA (Global Memory):** Massively parallel GPU computing.
3. **CUDA (Shared Memory):** Optimizing GPU memory bandwidth by caching image patches.


## Requirements
- **Compiler:** g++ (with OpenMP support)
- **CUDA Toolkit:** nvcc (NVIDIA CUDA Compiler)
- **Make:** Build system


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
| Flag |   Long Flag    |              Description               | Default |
|:---- |:--------------:|:--------------------------------------:| -------:|
| -m   |     --mode     |          Execution Mode <0-5>          |       0 |
| -n   |  --image-num   |   ID of the input image to use <0-3>   |       0 |
| -p   |  --patch-size  |     Size of the square patch (k×k)     |       5 |
| -f   | --filter-sigma | Smoothing parameter h (controls decay) |    0.06 |
| -s   | --patch-sigma  | Standard deviation for patch Gaussian  |     0.8 |
| -h   |     --help     |         Show usage information         |     N/A |

### Execution Modes (`-m`)
- `0` **CPU Serial**: Baseline single-threaded C++ implementation. Slowest, used for correctness verification.
- `1` **CPU Parallel**: Multi-threaded implementation using OpenMP. Parallelizes the outer loops.
- `2` **GPU Global Memory**: Naive CUDA implementation. Each thread computes one pixel, reading patches directly from Global Memory.
- `3` **GPU Global + Intrinsics:** Same as Mode 2, but uses hardware intrinsics (__expf, __fmaf_rn) for faster, lower-precision math.
- `4` **GPU Shared Memory**: Optimized CUDA implementation. Threads cooperatively load search windows into Shared Memory (L1 Cache) to minimize global memory bandwidth bottlenecks.
- `5` **GPU Shared + Intrinsics:** Combines Shared Memory caching with hardware intrinsics for maximum performance.

### Input Image (`-n`)
- `0` **House**: 64x64
- `1` **Flower**: 128x128
- `2` **Lena**: 256x256
- `3` **Lena**: 512x512


## Benchmarking
The project includes a shell script benchmark.sh to automate testing across multiple modes and image sizes. It handles warm-up runs, multiple iterations, and data export in `data/benchmark/benchmark_results.csv`.

### Syntax
```
./benchmark.sh [options]
```

### Options
| Flag |                   Description                    | Default |
|:---- |:------------------------------------------------:| -------:|
| -m   |   List of modes to run (quoted, e.g., `"2 3"`)   | 2 3 4 5 |
| -n   | List of image IDs to run (quoted, e.g., `"0 1"`) | 0 1 2 3 |
| -i   |          Number of iterations per test           |       5 |
| -t   |         Timer selection logic <1-2>              |       1 |

### Timer Selection (`-t`)
- `0` **NLM**: Only times the NLM portion.
- `1` **Total**: Time the total (serial + parallel) portion.

### Examples
1. **Fast GPU Comparison (Default):** Just run it. It defaults to modes 2, 3, 4, 5 (GPU only).
    ```
    ./benchmark.sh
    ```
2. **Full Test Run (CPU + GPU):** on image 0 and 1 only, with 3 iterations.
    ```
    ./benchmark.sh -m "0 1 2 3 4 5" -n "0 1" -i 3
    ```
3. **Testing Size Scaling:** Test only the Optimized GPU version (Mode 5) across all image sizes to see scalability.
    ```
    ./benchmark.sh -m "5" -n "0 1 2 3" -i 10
    ```


## Visualization
A Python script plot_results.py is provided to generate charts from the benchmark data.
Ensure you have generated benchmark_results.csv using the benchmark script, then run:
```
python3 plot_results.py
```
This will generate benchmark_plot.png, a bar chart visualizing the execution time (log scale).

### Example
<img width="600" alt="GPU_Optimizations_NLM" src="https://github.com/user-attachments/assets/41635718-9f54-46eb-9fbb-6c6e1e7b300c" />
