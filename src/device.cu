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
    for (int tile = 0; tile * tileDim <= dimension; tile++) {
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

