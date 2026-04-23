#pragma once
#include <cstddef>

void launchAddScalar(float *d_data, float scalar, size_t rows, size_t cols, cudaStream_t stream = 0);      // A = a + A
void launchSubtractScalar(float *d_data, float scalar, size_t rows, size_t cols, cudaStream_t stream = 0); // A = -a + A
void launchMultiplyScalar(float *d_data, float factor, size_t rows, size_t cols, cudaStream_t stream = 0); // A = a * A
void launchFusedScalarMultiplyAndAdd(float *d_data, float alpha, float beta, size_t rows, size_t cols, cudaStream_t stream = 0); // A = a * A + b

// Matrix-Vector Products
void launchDenseMatVec(const float* d_A, const float* d_x, float* d_y, size_t rows, size_t cols);
void launchCsrMatVec(const int* d_row_ptr, const int* d_col_ind, const float* d_val, 
                     const float* d_x, float* d_y, size_t num_rows);