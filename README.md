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

This is a massive improvement over CPU / single-threaded host code.
But how could can it get? What is the top standard? How well does cuBLAS do here?
Below is a comparison with the naive device implementation above.

| Dimension | Naive   | cuBLAS  |
|-----------|---------|---------|
| 32        | 0.2     | 0.5     |
| 512       | 1.5     | 1.25    |
| 2048      | 28      | 10      |
| 8192      | 1080    | 165     |

cuBLAS is higher for tiny matrices, likely due to library overhead, but on larger matrices, it crushes my naive implementation.
Speed improvement for cuBLAS on the largest matrix multiplication time is down 85%.
Let's optimize!

## 3. Device Shared Memory Multiplication

This is very similar to the earlier naive device implementation, except thread blocks collaboratively load 16x16 tiles into shared memory, sync. and then utilize shared memory for calculation.
For brevity. I am only including the 32 and 8192 dimensions.

### Performance Results

| Dimension | SharedM | Naive   | cuBLAS  |
|-----------|---------|---------|---------|
| 32        | 0.15    | 0.2     | 0.5     |
| 8192      | 770     | 1080    | 165     |

We have optimized down 29%, but we have a long way still to go.

## 4. Device Tensor Core Multiplication

NVIDIA GPUs include tensor cores with warp-wide instructions that can do matrix multiplication themselves.
Our next iteration is a shared memory + tensor core with warp-wide mma (matrix multiplication and addition).
This warp-wide instruction is available directly in the PTX ISA (wmma.mma).

### Performance Results

1. With 64x64 tiles, 16x64 slices loaded into shared memory, and 16x16x16 FP16 wmma, computation time is just 20% higher than cuBLAS. However, this leads to higher error than cuBLAS because it appears to preserve the full floating point computation. This comparison is not equal then.
2. With 32x32 tiles, 32x32 slices loaded into shared memory, and 16x16x8 TF32 wmma, memory reads should coalesce easily per warp. Interestingly after implementing, the error is almost identical to FP16. Additionally, latency has increased slightly. It appears that cuBLAS is not using tensor cores by default at all, and is using plain FP32 on cuda cores (crazy!) by default, and still going insanely fast. After manually enabling TF32 on cuBLAS, the error showed up the same as this implementation, and that latency barely decreased at all, which means for a 8192x8192 matrix, cuBLAS can effectively multiply it more accurately in ~ the same amount of time using FP32 instead of the fancy tensor cores. There is also a chance they are using double precision support on tensor.
