#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <x86intrin.h>
#include "host.cpp"
//#include "device.cu"

using namespace std;

random_device rd;
mt19937 randomAlgo(rd());
uniform_real_distribution<> dist(-256, 256);

void verifyMatrix(float *matrix, float *expectedMatrix, int dimension) {
    cout << "Verifying results..." << endl;
    float diffSum = 0;
    for (int row = 0; row < dimension; row++) {
        for (int col = 0; col < dimension; col++) {
            float expValue = expectedMatrix[row * dimension + col];
            float value = matrix[row * dimension + col];
            float diff = fabs(expValue - value);
            diffSum += diff;
        }
    }
    float avgDiff = diffSum / (dimension * dimension);
    cout << "avg diff: " << avgDiff << endl;
}

int main() {
    // build some large matrices here
    int dimension = 1024;

    /* DEVICE HEAP MEMORY */

    // define the raw data
    float *matrixA = new float[dimension * dimension];
    float *matrixB = new float[dimension * dimension];
    float *matrixHostC = new float[dimension * dimension];
    float *matrixDeviceC = new float[dimension * dimension];

    for (int row = 0; row < dimension; row++) {
        for (int col = 0; col < dimension; col++) {
            matrixA[row * dimension + col] = dist(randomAlgo);
            matrixB[row * dimension + col] = dist(randomAlgo);
        }
    }

    cout << "Generated matrices" << endl;

    cout << "Starting host matrix multiplication..." << endl;

    auto start_tp = std::chrono::high_resolution_clock::now();
    hostMatrixMultiply(matrixA, matrixB, matrixHostC, dimension);
    auto end_tp = std::chrono::high_resolution_clock::now();
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_tp - start_tp).count();
    cout << "Completed host matrix multiplication." << endl;
    cout << "Host multiply elapsed: " << (elapsed_ns / 1e6) << " ms" << endl;

    /*
    cout << "Starting naive device matrix multiplication..." << endl;
    start_tp = std::chrono::high_resolution_clock::now();
    naiveDeviceMatrixMultiply(matrixA, matrixB, matrixDeviceC, dimension);
    end_tp = std::chrono::high_resolution_clock::now();
    elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_tp - start_tp).count();
    cout << "Completed naive device matrix multiplication." << endl;
    cout << "Device multiply elapsed: " << (elapsed_ns / 1e6) << " ms" << endl;

    verifyMatrix(matrixDeviceC, matrixHostC, dimension);
    */

    // make sure we see gpu debug output
    cudaDeviceSynchronize();

    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixHostC;
    delete[] matrixDeviceC;

    return 0;
}
