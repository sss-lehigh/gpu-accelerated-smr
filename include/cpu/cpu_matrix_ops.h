#pragma once

#include <iostream>
#include <algorithm>
#include <cstddef>
#include <omp.h>
#include <cmath>

void addScalarCPU(float* __restrict__ data, float scalar, size_t rows, size_t cols) {
  size_t n = rows * cols;

  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < n; ++i) {
    data[i] += scalar;
  }
}

void subtractScalarCPU(float* __restrict__ data, float scalar, size_t rows, size_t cols) {
  size_t n = rows * cols;

  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < n; ++i) {
    data[i] -= scalar;
  }
}

void multiplyScalarCPU(float* __restrict__ data, float scalar, size_t rows, size_t cols) {
  size_t n = rows * cols;

  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < n; ++i) {
    data[i] *= scalar;
  }
}

void fusedScalarMultiplyAndAddCPU(float* __restrict__ data, float alpha, float beta, size_t rows, size_t cols) {
  size_t n = rows * cols;

  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < n; ++i) {
    data[i] = std::fma(data[i], alpha, beta);
  }
}

// C = A + B
void matrixAddCPU(const float* __restrict__ A, 
                  const float* __restrict__ B, 
                  float* __restrict__ C, 
                  size_t rows, size_t cols) {
  size_t n = rows * cols;

  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < n; ++i) {
    C[i] = A[i] + B[i];
  }
}

// C = A - B
void matrixSubCPU(const float* __restrict__ A, 
                  const float* __restrict__ B, 
                  float* __restrict__ C, 
                  size_t rows, size_t cols) {
  size_t n = rows * cols;

  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < n; ++i) {
    C[i] = A[i] - B[i];
  }
}

// A += B
void inPlaceMatrixAddCPU(float* __restrict__ A, 
                         const float* __restrict__ B, 
                         size_t rows, size_t cols) {
  size_t n = rows * cols;

  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < n; ++i) {
    A[i] += B[i];
  }
}

// A -= B
void inPlaceMatrixSubCPU(float* __restrict__ A, 
                         const float* __restrict__ B, 
                         size_t rows, size_t cols) {
  size_t n = rows * cols;

  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < n; ++i) {
    A[i] -= B[i];
  }
}

// Optimized CPU Matrix Multiplication (C = A * B)
void launchSgemmCPU(const float* __restrict__ A, 
                    const float* __restrict__ B, 
                    float* __restrict__ C, 
                    int M, int N, int K) {
                    
  // Initialize C to 0.0f
  // 'collapse(2)' merges the two loops into one large loop for OpenMP to divide
  #pragma omp parallel for simd collapse(2) schedule(static)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * N + j] = 0.0f;
    }
  }

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float a_ik = A[i * K + k]; 
      
      #pragma omp simd
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += a_ik * B[k * N + j];
      }
    }
  }
}

// C = A * B + D
void sgemmAddFusedCPU(const float* __restrict__ A, 
                      const float* __restrict__ B, 
                      const float* __restrict__ D,
                      float* __restrict__ C, 
                      int M, int N, int K) {
                      
  // Initialize C with the values of D.
  #pragma omp parallel for simd collapse(2) schedule(static)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * N + j] = D[i * N + j];
    }
  }

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float a_ik = A[i * K + k]; 
      
      #pragma omp simd
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += a_ik * B[k * N + j];
      }
    }
  }
}

// C = alpha * A + beta * B
void matrixScaleAddCPU(const float* __restrict__ A, 
                       const float* __restrict__ B, 
                       float* __restrict__ C, 
                       float alpha, float beta, 
                       size_t rows, size_t cols) {
  size_t n = rows * cols;

  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < n; ++i) {
    C[i] = std::fma(A[i], alpha, B[i] * beta);
  }
}

// C = alpha * (A * B) + beta * C
void sgemmInPlaceAccumulateCPU(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ C, 
                               int M, int N, int K,
                               float alpha, float beta) {
                               
  // beta * C
  #pragma omp parallel for simd collapse(2) schedule(static)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (beta == 0.0f) {
        C[i * N + j] = 0.0f;
      } else {
        C[i * N + j] *= beta;
      }
    }
  }

  // alpha * (A * B) + C.
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float a_ik_alpha = A[i * K + k] * alpha; 
      
      #pragma omp simd
      for (int j = 0; j < N; ++j) {
          C[i * N + j] += a_ik_alpha * B[k * N + j];
      }
    }
  }
}

// C = A .* B
void elementwiseMatrixMultCPU(const float* __restrict__ A, 
                              const float* __restrict__ B, 
                              float* __restrict__ C, 
                              size_t rows, size_t cols) {
  size_t n = rows * cols;

  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < n; ++i) {
    C[i] = A[i] * B[i];
  }
}

// A .*= B
void inPlaceElementwiseMatrixMultCPU(float* __restrict__ A, 
                                     const float* __restrict__ B, 
                                     size_t rows, size_t cols) {
  size_t n = rows * cols;

  #pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < n; ++i) {
    A[i] *= B[i];
  }
}