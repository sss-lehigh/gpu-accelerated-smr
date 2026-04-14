#pragma once
#include <cstddef>

void launchAddScalar(float *d_data, float scalar, size_t rows, size_t cols);
void launchScaleMatrix(float *d_data, float factor, size_t rows, size_t cols);
void launchScaleAndAdd(float *d_data, float factor, float scalar, size_t rows, size_t cols);

// Matrix-Vector Products
void launchDenseMatVec(const float* d_A, const float* d_x, float* d_y, size_t rows, size_t cols);
void launchCsrMatVec(const int* d_row_ptr, const int* d_col_ind, const float* d_val, 
                     const float* d_x, float* d_y, size_t num_rows);