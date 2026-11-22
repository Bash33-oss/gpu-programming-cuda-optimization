# GPU Programming and CUDA Optimization Portfolio

## Project Overview

This project demonstrates GPU programming and performance optimization using Python, CUDA, and Numba.

It implements several GPU-accelerated operations, including image processing, matrix multiplication (GEMM), and parallel reduction.

The project explores GPU kernel implementation, memory access patterns, shared-memory optimization, and performance benchmarking.

## Technologies Used

- Python
- NVIDIA CUDA
- Numba CUDA
- NumPy
- Google Colab
- NVIDIA Tesla T4 GPU

## Implemented Components

### 1. Image Processing and Computer Vision

- RGB-to-grayscale conversion using a custom CUDA kernel.
- 3×3 box blur using GPU parallel processing.
- CPU reference comparison for correctness validation.

### 2. Matrix Multiplication (GEMM)

- Custom GPU matrix multiplication kernel.
- Shared-memory tiled GEMM implementation.
- Comparison of naive and shared-memory implementations.
- Performance benchmarking.

### 3. Parallel Primitives

- GPU-based parallel sum reduction.
- Shared-memory utilization.
- Parallel computation and result validation.

## Performance Benchmarking

### GPU Environment

- GPU: NVIDIA Tesla T4
- Platform: Google Colab
- Programming Language: Python
- GPU Programming Framework: Numba CUDA

### GEMM Benchmark Results

Matrix dimensions: 512 × 512

| Implementation | Execution Time |
|---|---:|
| Naive GEMM | 6.5332 ms |
| Shared-memory GEMM | 5.5227 ms |

**Measured speedup:** 1.183×

**Performance improvement:** 15.48%

The shared-memory implementation achieved lower execution time than the naive implementation in this benchmark.

### Image Processing Results

| Operation | Execution Time |
|---|---:|
| Grayscale Conversion | 0.2275 ms |
| 3×3 Box Blur | 0.7167 ms |

Maximum grayscale pixel error compared with the CPU reference: 1.

*Benchmark results are from an individual test run and may vary depending on GPU load and execution conditions.*

## Correctness Validation

The following components were successfully executed:

- Grayscale conversion: SUCCESS
- 3×3 box blur: SUCCESS
- Matrix multiplication: SUCCESS
- Shared-memory GEMM: SUCCESS
- Parallel reduction: SUCCESS

GPU outputs were compared against reference computations where applicable.

## Optimization Approach

The project explores:

- GPU thread and block configuration.
- Parallel execution.
- Shared-memory tiling.
- Reduction operations.
- Kernel execution-time measurement.
- CPU versus GPU result validation.

## Project Objective

The objective is to demonstrate practical understanding of GPU kernel development, parallel computing, image processing, matrix operations, and GPU performance optimization.

## Future Improvements

- Benchmark additional matrix dimensions.
- Experiment with different CUDA block sizes.
- Optimize memory access patterns.
- Explore more advanced GEMM techniques.
- Compare additional GPU optimization strategies.

## Author

GPU Programming Portfolio Project

Developed and tested using Google Colab with an NVIDIA Tesla T4 GPU.

