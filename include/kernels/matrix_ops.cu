#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

#include "matrix_ops.h"
#include "common.cuh"

#define WARP_SIZE 32

// Kernel for shifting all matrix elements by a value
__global__ void addScalarVectorizedKernel(float* data, float scalar, size_t n) {
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

void launchAddScalar(float *d_data, float scalar, size_t rows, size_t cols) {
  size_t n = rows * cols;

  // typical block size. can be tuned
  int blockSize = 256;

  // Calculate grid size based on vectorized element count
  size_t n_vec = n / 4; 
  int desired_blocks = (n_vec + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 80 * 32); // Scaled for V100S SM count

  addScalarVectorizedKernel<<<num_blocks, blockSize>>>(d_data, scalar, n);
  CUDA_CHECK(cudaGetLastError());
}

// Kernel for scaling all matrix elements by a factor
__global__ void scaleMatrixGridStrideKernel(float* data, float factor, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = index; i < n; i += stride) {
    data[i] *= factor;
  }
}

void launchScaleMatrix(float *d_data, float factor, size_t rows, size_t cols) {
  size_t n = rows * cols;

  int blockSize = 256;

  // Calculate required blocks with a maximum limit of 32 * 256
  int desired_blocks = (n + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 32 * 256);

  scaleMatrixGridStrideKernel<<<num_blocks, blockSize>>>(d_data, factor, n);
  
  CUDA_CHECK(cudaGetLastError());
  
  CUDA_CHECK(cudaDeviceSynchronize()); // debugging (remove later)
}

// Kernel for fusing scaling and scalar addition
__global__ void scaleAndAddGridStrideKernel(float* data, float factor, float scalar, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = index; i < n; i += stride) {
    // fmaf(x, y, z) computes (x * y) + z as a single hardware instruction
    data[i] = fmaf(data[i], factor, scalar);
  }
}

void launchScaleAndAdd(float *d_data, float factor, float scalar, size_t rows, size_t cols) {
  size_t n = rows * cols;

  int blockSize = 256;

  int desired_blocks = (n + blockSize - 1) / blockSize;
  int num_blocks = std::min(desired_blocks, 32 * 256);

  scaleAndAddGridStrideKernel<<<num_blocks, blockSize>>>(d_data, factor, scalar, n);
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize()); // Remove later
}

//  Kernel for dense matrix-vector product (y = A * x)
__global__ void denseMatVecWarpKernel(const float* A, const float* x, float* y, size_t rows, size_t cols) {
  // Determine which row this warp is responsible for
  size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  if (warp_id < rows) {
    float sum = 0.0f;
        
    // Threads in the warp read consecutive elements in the row
    for (size_t col = lane; col < cols; col += WARP_SIZE) {
      sum += A[warp_id * cols + col] * x[col];
    }

    // Warp-level parallel reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 holds the final sum and writes it to the output vector
    if (lane == 0) {
      y[warp_id] = sum;
    }
  }
}

// Kernel for sparse matrix-vector product (CSR Format) (y = A * x)
__global__ void csrMatVecWarpKernel(const int* row_ptr, const int* col_ind, const float* val, 
                                    const float* x, float* y, size_t num_rows) {
  size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  if (warp_id < num_rows) {
    int row_start = row_ptr[warp_id];
    int row_end   = row_ptr[warp_id + 1];
    float sum = 0.0f;

    // Process non-zero elements of the sparse row
    for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
      sum += val[j] * x[col_ind[j]];
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
      y[warp_id] = sum;
    }
  }
}

void launchDenseMatVec(const float* d_A, const float* d_x, float* d_y, size_t rows, size_t cols) {
  int blockSize = 256; 
  int warpsPerBlock = blockSize / WARP_SIZE; // 256 / 32 = 8 warps per block

  // We need one warp per row. Calculate how many blocks we need to get 'rows' warps.
  int numBlocks = (rows + warpsPerBlock - 1) / warpsPerBlock;

  denseMatVecWarpKernel<<<numBlocks, blockSize>>>(d_A, d_x, d_y, rows, cols);
  
  CUDA_CHECK(cudaGetLastError());
}

void launchCsrMatVec(const int* d_row_ptr, const int* d_col_ind, const float* d_val, 
                     const float* d_x, float* d_y, size_t num_rows) {
  int blockSize = 256; 
  int warpsPerBlock = blockSize / WARP_SIZE; 

  int numBlocks = (num_rows + warpsPerBlock - 1) / warpsPerBlock;

  csrMatVecWarpKernel<<<numBlocks, blockSize>>>(d_row_ptr, d_col_ind, d_val, d_x, d_y, num_rows);
  
  CUDA_CHECK(cudaGetLastError());
}