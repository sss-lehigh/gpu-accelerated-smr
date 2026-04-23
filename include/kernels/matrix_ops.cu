#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

#include "matrix_ops.h"
#include "common.cuh"

#define WARP_SIZE 32
#define TILE_SIZE 32

// Kernel for shifting all matrix elements by a value, A = a + A
__global__ void addScalarKernel(float* data, float scalar, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  size_t n_vec = n / 4;
  float4 *data_vec = reinterpret_cast<float4 *>(data);
  float4 scalar_vec = make_float4(scalar, scalar, scalar, scalar);

  for (size_t i = index; i < n_vec; i+= stride) {
    float4 val = data_vec[i];
    val.x += scalar_vec.x;
    val.y += scalar_vec.y;
    val.z += scalar_vec.z;
    val.w += scalar_vec.w;
    data_vec[i] = val;
  }

  // Handle the tail elements (0 to 3 elements)
  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start + index; i < n; i += stride) {
    data[i] += scalar;
  }
}

void launchAddScalar(float *d_data, float scalar, size_t rows, size_t cols, cudaStream_t stream = 0) {
  size_t n = rows * cols;

  // typical block size. can be tuned
  int blockSize = 256;

  // Calculate grid size based on vectorized element count
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32); // Scaled for V100S SM count

  addScalarKernel<<<num_blocks, blockSize>>>(d_data, scalar, n);
  CUDA_CHECK(cudaGetLastError());
}

void launchSubtractScalar(float *d_data, float scalar, size_t rows, size_t cols, cudaStream_t stream = 0) {
  size_t n = rows * cols;
  scalar = (-1.0f)* scalar;

  // typical block size. can be tuned
  int blockSize = 256;

  // Calculate grid size based on vectorized element count
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32); // Scaled for V100S SM count

  addScalarKernel<<<num_blocks, blockSize>>>(d_data, scalar, n);
  CUDA_CHECK(cudaGetLastError());
}

// Vectorized kernel for matrix scaling
__global__ void multiplyScalarKernel(float* data, float factor, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  float4 *data_vec = reinterpret_cast<float4 *>(data);

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = index; i < n_vec; i += stride) {
    float4 val = data_vec[i];
    val.x *= factor;
    val.y *= factor;
    val.z *= factor;
    val.w *= factor;
    data_vec[i] = val;
  }

  // Tail processing for remaining elements (0 to 3 floats)
  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start + index; i < n; i += stride) {
    data[i] *= factor;
  }
}

void launchMultiplyScalar(float *d_data, float factor, size_t rows, size_t cols, cudaStream_t stream = 0) {
  size_t n = rows * cols;

  int blockSize = 256;

  // Calculate required blocks based on the vectorized count, not the float count
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32);

  multiplyScalarKernel<<<num_blocks, blockSize>>>(d_data, factor, n);
  
  CUDA_CHECK(cudaGetLastError());
}

// Vectorized kernel for fused multiply-add (y = alpha * x + beta)
__global__ void fusedSclarMultiplyAndAddKernel(float* data, float alpha, float beta, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  float4 *data_vec = reinterpret_cast<float4 *>(data);

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = index; i < n_vec; i += stride) {
    float4 val = data_vec[i];
    
    // Hardware Fused Multiply-Add (FMA)
    val.x = fmaf(val.x, alpha, beta);
    val.y = fmaf(val.y, alpha, beta);
    val.z = fmaf(val.z, alpha, beta);
    val.w = fmaf(val.w, alpha, beta);
    
    data_vec[i] = val;
  }

  // Tail processing
  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start + index; i < n; i += stride) {
    data[i] = fmaf(data[i], alpha, beta);
  }
}

void launchFusedScalarMultiplyAndAdd(float *d_data, float alpha, float beta, size_t rows, size_t cols, cudaStream_t stream = 0) {
  size_t n = rows * cols;
  int blockSize = 256;

  // Grid configuration based on 128-bit vectorized chunks
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32); 

  fusedSclarMultiplyAndAddKernel<<<num_blocks, blockSize>>>(d_data, alpha, beta, n);
  CUDA_CHECK(cudaGetLastError());
}

// Vectorized kernel for matrix-matrix addition (C = A + B)
__global__ void matrixAddVectorizedKernel(const float* A, const float* B, float* C, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  
  // Cast pointers to float4 for 128-bit memory transactions.
  // A, B, and C must all be 16-byte aligned.
  const float4* A_vec = reinterpret_cast<const float4*>(A);
  const float4* B_vec = reinterpret_cast<const float4*>(B);
  float4* C_vec = reinterpret_cast<float4*>(C);

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = index; i < n_vec; i += stride) {
    float4 a = A_vec[i];
    float4 b = B_vec[i];
    
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    
    // Single 128-bit store
    C_vec[i] = c;
  }

  // Tail processing for arrays not perfectly divisible by 4
  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start + index; i < n; i += stride) {
    C[i] = A[i] + B[i];
  }
}

void launchMatrixAdd(const float *d_A, const float *d_B, float *d_C, size_t rows, size_t cols, cudaStream_t stream = 0) {
  size_t n = rows * cols;
  int blockSize = 256;

  // Calculate grid size based on 128-bit chunks
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32); 

  matrixAddVectorizedKernel<<<num_blocks, blockSize>>>(d_A, d_B, d_C, n);
  CUDA_CHECK(cudaGetLastError());
}

// Vectorized kernel for in-place matrix addition (A += B)
__global__ void inPlaceMatrixAddVectorizedKernel(float* __restrict__ A, const float* __restrict__ B, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  
  // Cast pointers to float4 for 128-bit memory transactions.
  float4* A_vec = reinterpret_cast<float4*>(A);
  const float4* B_vec = reinterpret_cast<const float4*>(B);

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = index; i < n_vec; i += stride) {
    // Load 128-bits from A and B
    float4 a = A_vec[i];
    float4 b = B_vec[i];
    
    // Perform vector addition locally in registers
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    
    // Store 128-bits back into A
    A_vec[i] = a;
  }

  // Tail processing for arrays not perfectly divisible by 4
  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start + index; i < n; i += stride) {
    A[i] += B[i];
  }
}

void launchInPlaceMatrixAdd(float *d_A, const float *d_B, size_t rows, size_t cols, cudaStream_t stream = 0) {
  size_t n = rows * cols;
  int blockSize = 256;

  // Calculate grid size based on 128-bit chunks
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32); 

  inPlaceMatrixAddVectorizedKernel<<<num_blocks, blockSize>>>(d_A, d_B, n);
  CUDA_CHECK(cudaGetLastError());
}

// Vectorized kernel for matrix-matrix subtraction (C = A - B)
__global__ void matrixSubVectorizedKernel(const float* __restrict__ A, 
                                          const float* __restrict__ B, 
                                          float* __restrict__ C, 
                                          size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  
  // Cast pointers to float4 for 128-bit memory transactions.
  // A, B, and C must be 16-byte aligned.
  const float4* A_vec = reinterpret_cast<const float4*>(A);
  const float4* B_vec = reinterpret_cast<const float4*>(B);
  float4* C_vec = reinterpret_cast<float4*>(C);

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = index; i < n_vec; i += stride) {
    float4 a = A_vec[i];
    float4 b = B_vec[i];
    
    float4 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;
    c.w = a.w - b.w;
    
    // Single 128-bit store
    C_vec[i] = c;
  }

  // Tail processing for arrays not perfectly divisible by 4
  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start + index; i < n; i += stride) {
    C[i] = A[i] - B[i];
  }
}

void launchMatrixSub(const float *d_A, const float *d_B, float *d_C, size_t rows, size_t cols, cudaStream_t stream = 0) {
  size_t n = rows * cols;
  int blockSize = 256;

  // Calculate grid size based on 128-bit chunks
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32); 

  matrixSubVectorizedKernel<<<num_blocks, blockSize>>>(d_A, d_B, d_C, n);
  CUDA_CHECK(cudaGetLastError());
}

// Vectorized kernel for in-place matrix subtraction (A -= B)
__global__ void inPlaceMatrixSubVectorizedKernel(float* __restrict__ A, 
                                                 const float* __restrict__ B, 
                                                 size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  
  // Cast pointers to float4 for 128-bit memory transactions.
  // A is mutable (read/write), B is read-only.
  float4* A_vec = reinterpret_cast<float4*>(A);
  const float4* B_vec = reinterpret_cast<const float4*>(B);

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = index; i < n_vec; i += stride) {
    float4 a = A_vec[i];
    float4 b = B_vec[i];
    
    // Perform vector subtraction locally in registers
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    
    // Store the 128-bit result back into A
    A_vec[i] = a;
  }

  // Tail processing for arrays not perfectly divisible by 4
  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start + index; i < n; i += stride) {
    A[i] -= B[i];
  }
}

void launchInPlaceMatrixSub(float *d_A, const float *d_B, size_t rows, size_t cols, cudaStream_t stream = 0) {
  size_t n = rows * cols;
  int blockSize = 256;

  // Calculate grid size based on 128-bit chunks
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32); 

  inPlaceMatrixSubVectorizedKernel<<<num_blocks, blockSize>>>(d_A, d_B, n);
  CUDA_CHECK(cudaGetLastError());
}

// GEMM using Shared Memory Tiling
__global__ void sgemmSharedMemoryTiledKernel(const float* __restrict__ A, 
                                             const float* __restrict__ B, 
                                             float* __restrict__ C, 
                                             int M, int N, int K) {
  // Allocate static shared memory for the tiles of A and B
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  // Calculate global row and column indices for the C matrix
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Local register to accumulate the dot product
  float sum = 0.0f;

  // Loop over the K dimension in steps of TILE_SIZE
  int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
  for (int t = 0; t < numTiles; ++t) {
    
    // Load data from global to shared memory.
    // Include bounds checking for matrices whose dimensions are not multiples of TILE_SIZE.
    if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f; // Zero-pad out-of-bounds
    }

    if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f; // Zero-pad out-of-bounds
    }

    // Synchronize to ensure all threads in the block have finished loading the tile
    __syncthreads();

    // Compute the partial dot product for the current tile
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }

    // Synchronize again to ensure all threads are done computing before 
    // the next iteration overwrites the shared memory tiles
    __syncthreads();
  }

  // Write the final accumulated sum to global memory
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

void launchSgemm(const float *d_A, const float *d_B, float *d_C, int M, int N, int K, cudaStream_t stream = 0) {
  // Use a 2D block topology mapping directly to the tile size
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  
  // Calculate grid dimensions to cover the entire M x N output matrix
  dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);

  sgemmSharedMemoryTiledKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

// Fused GEMM + Matrix Addition = (A x B) + D
__global__ void sgemmAddFusedKernel(const float* __restrict__ A, 
                                    const float* __restrict__ B, 
                                    const float* __restrict__ D,
                                    float* __restrict__ C, 
                                    int M, int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;

  int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
  for (int t = 0; t < numTiles; ++t) {
    // Collaborative load of A tile into shared memory
    if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f; 
    }

    // Collaborative load of B tile into shared memory
    if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f; 
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }
    __syncthreads();
  }

  // Add the D matrix
  if (row < M && col < N) {
    int idx = row * N + col;

    // Read D once, add to the register, then store to C
    C[idx] = sum + D[idx];
  }
}

void launchSgemmAddFused(const float *d_A, const float *d_B, const float *d_D, float *d_C, 
                         int M, int N, int K, cudaStream_t stream = 0) {
                         
  // Define a 2D block matching the tile dimensions.
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  
  // Calculate grid dimensions to ensure we launch enough blocks to cover the output matrix C.
  dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);

  sgemmAddFusedKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_D, d_C, M, N, K);
  
  CUDA_CHECK(cudaGetLastError());
}

// Fused Matrix Addition and Scaling: C = (alpha * A) + (beta * B)
__global__ void matrixScaleAddVectorizedKernel(const float* __restrict__ A, 
                                               const float* __restrict__ B, 
                                               float* __restrict__ C, 
                                               float alpha, float beta, 
                                               size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t n_vec = n / 4;
  
  const float4* A_vec = reinterpret_cast<const float4*>(A);
  const float4* B_vec = reinterpret_cast<const float4*>(B);
  float4* C_vec = reinterpret_cast<float4*>(C);

  for (size_t i = index; i < n_vec; i += stride) {
    float4 a = A_vec[i];
    float4 b = B_vec[i];
    float4 c;
    
    // Using FMA (Fused Multiply-Add) to execute
    c.x = fmaf(a.x, alpha, b.x * beta);
    c.y = fmaf(a.y, alpha, b.y * beta);
    c.z = fmaf(a.z, alpha, b.z * beta);
    c.w = fmaf(a.w, alpha, b.w * beta);
    
    C_vec[i] = c;
  }

  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start + index; i < n; i += stride) {
    C[i] = fmaf(A[i], alpha, B[i] * beta);
  }
}

void launchMatrixScaleAdd(const float *d_A, const float *d_B, float *d_C, 
                          float alpha, float beta, size_t rows, size_t cols, cudaStream_t stream = 0) {
  size_t n = rows * cols;
  int blockSize = 256;
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32); 

  // Launch the kernel
  matrixScaleAddVectorizedKernel<<<num_blocks, blockSize>>>(d_A, d_B, d_C, alpha, beta, n);
  
  CUDA_CHECK(cudaGetLastError());
}

// Optimized GEMM with In-Place Accumulation (C = alpha * (A * B) + beta * C)
__global__ void sgemmInPlaceAccumulateKernel(const float* __restrict__ A, 
                                             const float* __restrict__ B, 
                                             float* __restrict__ C, 
                                             int M, int N, int K,
                                             float alpha, float beta) {
  
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f;

  int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
  for (int t = 0; t < numTiles; ++t) {
    
    // Load tiles into shared memory
    if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot product
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }

    __syncthreads();
  }

  // Write the accumulated sum back to global memory IN PLACE
  if (row < M && col < N) {
    int idx = row * N + col;
    // Scale the dot product by alpha, add it to the scaled existing C value
    C[idx] = alpha * sum + beta * C[idx];
  }
}

void launchSgemmAccumulate(const float *d_A, const float *d_B, float *d_C, 
                           int M, int N, int K, float alpha = 1.0f, float beta = 1.0f, cudaStream_t stream = 0) {
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);

  sgemmInPlaceAccumulateKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
  CUDA_CHECK(cudaGetLastError());
}

// Vectorized kernel for elementwies matrix-matrix multiplication (C = A . B)
__global__ void matrixElementwiseMultVectorizedKernel(const float* A, const float* B, float* C, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  
  // Cast pointers to float4 for 128-bit memory transactions.
  // A, B, and C must all be 16-byte aligned.
  const float4* A_vec = reinterpret_cast<const float4*>(A);
  const float4* B_vec = reinterpret_cast<const float4*>(B);
  float4* C_vec = reinterpret_cast<float4*>(C);

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = index; i < n_vec; i += stride) {
    float4 a = A_vec[i];
    float4 b = B_vec[i];
    
    float4 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.z = a.z * b.z;
    c.w = a.w * b.w;
    
    // Single 128-bit store
    C_vec[i] = c;
  }

  // Tail processing for arrays not perfectly divisible by 4
  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start + index; i < n; i += stride) {
    C[i] = A[i] * B[i];
  }
}

void launchElementwiseMatrixMult(const float *d_A, const float *d_B, float *d_C, size_t rows, size_t cols, cudaStream_t stream = 0) {
  size_t n = rows * cols;
  int blockSize = 256;

  // Calculate grid size based on 128-bit chunks
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32); 

  matrixElementwiseMultVectorizedKernel<<<num_blocks, blockSize>>>(d_A, d_B, d_C, n);
  CUDA_CHECK(cudaGetLastError());
}

// Vectorized kernel for in-place element wise matrix multiplication (A .= B)
__global__ void inPlaceMatrixElementwiseMultVectorizedKernel(float* __restrict__ A, const float* __restrict__ B, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  
  // Cast pointers to float4 for 128-bit memory transactions.
  float4* A_vec = reinterpret_cast<float4*>(A);
  const float4* B_vec = reinterpret_cast<const float4*>(B);

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = index; i < n_vec; i += stride) {
    // Load 128-bits from A and B
    float4 a = A_vec[i];
    float4 b = B_vec[i];
    
    // Perform vector addition locally in registers
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    
    // Store 128-bits back into A
    A_vec[i] = a;
  }

  // Tail processing for arrays not perfectly divisible by 4
  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start + index; i < n; i += stride) {
    A[i] *= B[i];
  }
}

void launchInPlaceMatrixElementwiseMultVectorizedKernel(float *d_A, const float *d_B, size_t rows, size_t cols, cudaStream_t stream = 0) {
  size_t n = rows * cols;
  int blockSize = 256;

  // Calculate grid size based on 128-bit chunks
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32); 

  inPlaceMatrixElementwiseMultVectorizedKernel<<<num_blocks, blockSize>>>(d_A, d_B, n);
  CUDA_CHECK(cudaGetLastError());
}