#pragma once
#include <cstddef>

// Scalar operations
// A = a + A
void launchAddScalar(float *d_data, float scalar, size_t rows, size_t cols,
                     cudaStream_t stream = 0);
// A = -a + A
void launchSubtractScalar(float *d_data, float scalar, size_t rows, size_t cols,
                          cudaStream_t stream = 0);
// A = a * A
void launchMultiplyScalar(float *d_data, float factor, size_t rows, size_t cols,
                          cudaStream_t stream = 0);
// A = a * A + b
void launchFusedScalarMultiplyAndAdd(float *d_data, float alpha, float beta, size_t rows,
                                     size_t cols, cudaStream_t stream = 0);

// Elementwise Matrix Matrix Operations
// C = A + B
void launchMatrixAdd(const float *d_A, const float *d_B, float *d_C, size_t rows,
                     size_t cols, cudaStream_t stream = 0);
// A += B
void launchInPlaceMatrixAdd(float *d_A, const float *d_B, size_t rows, size_t cols,
                            cudaStream_t stream = 0);
// C = A - B
void launchMatrixSub(const float *d_A, const float *d_B, float *d_C, size_t rows,
                     size_t cols, cudaStream_t stream = 0);
// A -= B
void launchInPlaceMatrixSub(float *d_A, const float *d_B, size_t rows, size_t cols,
                            cudaStream_t stream = 0);
// C = a * A + b * B
void launchFusedScalarMultMatrixAdd(const float *d_A, const float *d_B, float *d_C,
                                    float alpha, float beta, size_t rows, size_t cols,
                                    cudaStream_t stream = 0);
// C = A . B
void launchElementwiseMatrixMult(const float *d_A, const float *d_B, float *d_C,
                                 size_t rows, size_t cols, cudaStream_t stream = 0);
// A .= B
void launchInPlaceElementwiseMatrixMult(float *d_A, const float *d_B, size_t rows,
                                        size_t cols, cudaStream_t stream = 0);

// Matrix Matrix Operations
// C = A x B
void launchSgemm(const float *d_A, const float *d_B, float *d_C, int M, int N, int K,
                 cudaStream_t stream = 0);
// C = A x B + D
void launchFusedSgemmAdd(const float *d_A, const float *d_B, const float *d_D,
                         float *d_C, int M, int N, int K, cudaStream_t stream = 0);
// C = a * (A x B) + b * C
void launchInPlaceSgemmAccumulate(const float *d_A, const float *d_B, float *d_C, int M,
                                  int N, int K, float alpha = 1.0f, float beta = 1.0f,
                                  cudaStream_t stream = 0);