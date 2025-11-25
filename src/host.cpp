/**
 * This is a naive host matrix multiplication implementation for verification
 *
 * Boring old CPU code, also interesting for comparison of performance
 */
void hostMatrixMultiply(float *A, float *B, float *C, int dimension) {
    for (int row = 0; row < dimension; row++) {
        for (int col = 0; col < dimension; col++) {
            C[row * dimension + col] = 0;
            for (int k = 0; k < dimension; k++) {
                C[row * dimension + col] +=
                    A[row * dimension + k] * B[k * dimension + col];
            }
        }
    }
}

