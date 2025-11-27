#include <mma.h>

using namespace std;

using namespace nvcuda;

__device__ float getElement(float *matrix, int row, int col, int dimension) {
    if (row >= 0 && col >= 0 && row < dimension && col < dimension) {
        return matrix[row * dimension + col];
    } else {
        return 0;
    }
}

__device__ float getSubElement(float *matrix, int baseRow, int baseCol, int rowOffset, int colOffset, int dimension) {
    int row = baseRow + rowOffset;
    int col = baseCol + colOffset;
    return getElement(matrix, row, col, dimension);
}

/**
 * The most basic device matrix multiplication kernel
 *
 * Every memory access is global memory access, every thread computes one
 * element of the output matrix and does not help out other threads
 */
__global__ void d_naiveDeviceMatrixMultiply(float *A, float *B, float *C, int dimension) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dimension && col < dimension) {
        float value = 0;
        for (int k = 0; k < dimension; k++) {
            value += getElement(A, row, k, dimension) * getElement(B, k, col, dimension);
        }
        C[row * dimension + col] = value;
    }
}

__global__ void d_memOptDeviceMatrixMultiply(float *A, float *B, float *C, int dimension) {
    int baseRow = blockIdx.y * blockDim.y;
    int baseCol = blockIdx.x * blockDim.x;
    int row = baseRow + threadIdx.y;
    int col = baseCol + threadIdx.x;

    const int tileDim = 16;
    // load into shared memory
    __shared__ float sharedA[tileDim][tileDim];
    __shared__ float sharedB[tileDim][tileDim];

    // For matrices not a multiple of tileDim may still need threads outside the matrix
    // because they contribute intermediate tiles
    float value = 0;
    for (int tile = 0; tile * tileDim < dimension; tile++) {
        sharedA[threadIdx.y][threadIdx.x] = getSubElement(A, baseRow, tile * tileDim, threadIdx.y, threadIdx.x, dimension);
        sharedB[threadIdx.y][threadIdx.x] = getSubElement(B, tile * tileDim, baseCol, threadIdx.y, threadIdx.x, dimension);

        __syncthreads();

        for (int k = 0; k < tileDim; k++) {
            value += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < dimension && col < dimension) {
        C[row * dimension + col] = value;
    }
}

__global__ void d_TensorDeviceMatrixMultiply(float *A, float *B, float *C, int dimension) {
    // a single warp will handle a 16x16 tile
    // a thread block will handle a 64x64 tile
    // so we have 4x4=16 warps per block
    // threads are 1-dimensional: in a block 4x4x32=512 threads

    const int tileDim = 16;
    int baseRow = blockIdx.y * 64;
    int baseCol = blockIdx.x * 64;
    int warpId = threadIdx.x / 32;
    int fragBaseRow = (warpId / 4) * tileDim;
    int fragBaseCol = (warpId % 4) * tileDim;

    // load into shared memory
    __shared__ half sharedA[64][tileDim];
    __shared__ half sharedB[tileDim][64];

    // a and b are FP16 and c is FP32 .. this is the best you can do and still get 16x16x16
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_tile;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_tile;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_tile;

    // this fragment will live in registers until written to global memory
    wmma::fill_fragment(c_tile, 0.0f);

    for (int tile = 0; tile * tileDim < dimension; tile++) {
        for (int memIdx = threadIdx.x; memIdx < 64 * 16; memIdx += blockDim.x) {
            int tileRow, tileCol;

            // slice of A
            tileRow = memIdx / 16;
            tileCol = memIdx % 16;
            sharedA[tileRow][tileCol] = getSubElement(A, baseRow, tile * 16, tileRow, tileCol, dimension);

            // slice of B
            tileRow = memIdx / 64;
            tileCol = memIdx % 64;
            sharedB[tileRow][tileCol] = getSubElement(B, tile * 16, baseCol, tileRow, tileCol, dimension);
        }

        __syncthreads();

        load_matrix_sync(a_tile, &sharedA[fragBaseRow][0], 16);
        load_matrix_sync(b_tile, &sharedB[0][fragBaseCol], 64);

        // mma: C = A x B + C
        wmma::mma_sync(c_tile, a_tile, b_tile, c_tile);
        __syncthreads();
    }

    float* const globalCPtr = &C[(baseRow + fragBaseRow) * dimension + baseCol + fragBaseCol];
    store_matrix_sync(globalCPtr, c_tile, dimension, wmma::mem_row_major);
}

__global__ void d_TensorTf32DeviceMatrixMultiply(float *A, float *B, float *C, int dimension) {
    // a single warp will handle a 16x16 tile
    // a thread block will handle a 32x32 tile
    // so we have 2x2=4 warps per block
    // threads are 1-dimensional: in a block 2x2x32=128 threads

    int baseRow = blockIdx.y * 32;
    int baseCol = blockIdx.x * 32;
    int warpId = threadIdx.x / 32;
    int fragBaseRow = (warpId / 2) * 16;
    int fragBaseCol = (warpId % 2) * 16;

    // load into shared memory
    __shared__ float sharedA[32][32];
    __shared__ float sharedB[32][32];

    // a and b are FP16 and c is FP32 .. this is the best you can do and still get 16x16x16
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_tile;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_tile;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_tile;

    // this fragment will live in registers until written to global memory
    wmma::fill_fragment(c_tile, 0.0f);

    for (int tile = 0; tile * 32 < dimension; tile++) {
        for (int memIdx = threadIdx.x; memIdx < 32 * 32; memIdx += blockDim.x) {
            int tileRow, tileCol;

            // slice of A
            tileRow = memIdx / 32;
            tileCol = memIdx % 32;
            sharedA[tileRow][tileCol] = wmma::__float_to_tf32(getSubElement(A, baseRow, tile * 32, tileRow, tileCol, dimension));
            sharedB[tileRow][tileCol] = wmma::__float_to_tf32(getSubElement(B, tile * 32, baseCol, tileRow, tileCol, dimension));
        }

        __syncthreads();

        load_matrix_sync(a_tile, &sharedA[fragBaseRow][0], 32);
        load_matrix_sync(b_tile, &sharedB[0][fragBaseCol], 32);
        wmma::mma_sync(c_tile, a_tile, b_tile, c_tile);

        load_matrix_sync(a_tile, &sharedA[fragBaseRow][8], 32);
        load_matrix_sync(b_tile, &sharedB[8][fragBaseCol], 32);
        wmma::mma_sync(c_tile, a_tile, b_tile, c_tile);

        load_matrix_sync(a_tile, &sharedA[fragBaseRow][16], 32);
        load_matrix_sync(b_tile, &sharedB[16][fragBaseCol], 32);
        wmma::mma_sync(c_tile, a_tile, b_tile, c_tile);

        load_matrix_sync(a_tile, &sharedA[fragBaseRow][24], 32);
        load_matrix_sync(b_tile, &sharedB[24][fragBaseCol], 32);
        wmma::mma_sync(c_tile, a_tile, b_tile, c_tile);

        __syncthreads();
    }

    float* const globalCPtr = &C[(baseRow + fragBaseRow) * dimension + baseCol + fragBaseCol];
    store_matrix_sync(globalCPtr, c_tile, dimension, wmma::mem_row_major);
}

void mallocKernelMatrices(float **d_A, float **d_B, float **d_C, float *A, float *B, int dimension) {
    cudaMalloc(d_A, dimension * dimension * sizeof(float));
    cudaMalloc(d_B, dimension * dimension * sizeof(float));
    cudaMalloc(d_C, dimension * dimension * sizeof(float));

    cudaMemcpy(*d_A, A, dimension * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_B, B, dimension * dimension * sizeof(float), cudaMemcpyHostToDevice);
}

void freeKernelMatrices(float *d_A, float *d_B, float *d_C, float *C, int dimension) {
    cudaMemcpy(C, d_C, dimension * dimension * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void naiveDeviceMatrixMultiply(float *A, float *B, float *C, int dimension) {
    float *d_A, *d_B, *d_C;
    mallocKernelMatrices(&d_A, &d_B, &d_C, A, B, dimension);

    int blockDimension = 16;
    dim3 blockSize(blockDimension, blockDimension);
    int gridDimension = (dimension + blockDimension - 1) / blockDimension;
    dim3 gridSize(gridDimension, gridDimension);

    d_naiveDeviceMatrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, dimension);

    freeKernelMatrices(d_A, d_B, d_C, C, dimension);
}

void memOptDeviceMatrixMultiply(float *A, float *B, float *C, int dimension) {
    float *d_A, *d_B, *d_C;
    mallocKernelMatrices(&d_A, &d_B, &d_C, A, B, dimension);

    int blockDimension = 16;
    dim3 blockSize(blockDimension, blockDimension);
    int gridDimension = (dimension + blockDimension - 1) / blockDimension;
    dim3 gridSize(gridDimension, gridDimension);

    d_memOptDeviceMatrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, dimension);

    freeKernelMatrices(d_A, d_B, d_C, C, dimension);
}

void tensorDeviceMatrixMultiply(float *A, float *B, float *C, int dimension) {
    float *d_A, *d_B, *d_C;
    mallocKernelMatrices(&d_A, &d_B, &d_C, A, B, dimension);

    if (dimension % 64 != 0) {
        std::cerr << "Tensor core matrix multiply requires dimensions to be multiple of 64" << std::endl;
        exit(1);
    }

    dim3 blockSize(4 * 4 * 32); // 4x4 warps of 32 threads each
    dim3 gridSize(dimension / 64, dimension / 64);

    d_TensorDeviceMatrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, dimension);

    freeKernelMatrices(d_A, d_B, d_C, C, dimension);
}

void tensorTf32DeviceMatrixMultiply(float *A, float *B, float *C, int dimension) {
    float *d_A, *d_B, *d_C;
    mallocKernelMatrices(&d_A, &d_B, &d_C, A, B, dimension);

    if (dimension % 32 != 0) {
        std::cerr << "Tensor core matrix multiply requires dimensions to be multiple of 64" << std::endl;
        exit(1);
    }

    dim3 blockSize(2 * 2 * 32); // 2x2 warps of 32 threads each
    dim3 gridSize(dimension / 32, dimension / 32);

    d_TensorTf32DeviceMatrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, dimension);

    freeKernelMatrices(d_A, d_B, d_C, C, dimension);
}
