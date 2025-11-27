#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace std;

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " : ";
        cerr << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

void checkCublas(cublasStatus_t stat, const char* msg) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "cuBLAS Error: " << msg << endl;
        exit(1);
    }
}

void cublassMatrixMultiply(float *A, float *B, float *C, int dimension) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, dimension * dimension * sizeof(float));
    cudaMalloc(&d_B, dimension * dimension * sizeof(float));
    cudaMalloc(&d_C, dimension * dimension * sizeof(float));

    cudaMemcpy(d_A, A, dimension * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, dimension * dimension * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    // cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH); // forces true FP32

    cublasMath_t mode;
    cublasGetMathMode(handle, &mode);
    cout << "cuBLAS math mode: " << (int)mode << endl;
    // cuBLAS uses column-major, and this is row-major
    // from linear algebra C = A x B is the same as C^T = B^T x A^T

    // ie: (row-major)C = (row-major)A x (row-major)B is the same as
    // (column-major)C = (column-major)B x (column-major)A <= perfect for cuBLAS!

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N, // actually B^T, bc passing in row-major
        CUBLAS_OP_N, // actually A^T, bc passing in row-major
        dimension,
        dimension,
        dimension,
        &alpha,
        d_B, dimension, // B^T x A^T
        d_A, dimension,
        &beta,
        d_C, dimension); // output is C^T col-major, or C row-major

    cudaMemcpy(C, d_C, dimension * dimension * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
