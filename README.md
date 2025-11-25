# CUDA Matrix Multiplication Optimization

The goal of this project is to match (or beat) cuBLAS in matrix multiplication performance. The multiplication must be on large, square, single floating point matrices, as the primary concern is not to address edge cases, but to simply optimize extremely large multiplications.

## 1. CPU/Host Multiplication

Initial code uses serial (single-threaded) CPU code to perform the calculation.

### Performance Results

| Dimension | Time (ms) |
|-----------|-----------|
| 64        | 1.5       |
| 256       | 53        |
| 512       | 430       |
| 1024      | 4200      |

At even a modest size of 1024x1024, CPU host multiplication is already at 4200ms.

## 2. Device Naive Multiplication

Device code that does not optimize on GPU shared memory.
All threads behave individually and pull from global memory.

### Performance Results

| Dimension | Time (ms) |
|-----------|-----------|
| 512       | 1.5       |
| 1024      | 5         |
| 2048      | 28        |
| 4096      | 165       |
| 8192      | 1080      |

This is a massive improvement over CPU / sing-threaded host code.
But how could can it get? What is the top standard? How well does cuBLAS do here?




