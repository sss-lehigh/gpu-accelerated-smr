#pragma once
#include <iostream>
#include <algorithm>
#include <cstddef>


#define WARP_SIZE 32
#define TILE_SIZE 32
#define thread_count 8

struct float4 {
    float x, y, z, w;
    float4(float i, float j, float k, float l) : x(i), y(j), z(k), w(l) {}
    float4() : x(0), y(0), z(0), w(0) {}
    static void create_float4_vec(float4 *ret, float *data) {
        uint64_t i = 0;
        while (data != nullptr) {
            ret[i] = float4{data[0], data[1], data[2], data[3]};
            i += 1;
            data += 4;
        }
    }

    static void create_float4_vec(float4 *ret, const float *data) {
        uint64_t i = 0;
        while (data != nullptr) {
            ret[i] = float4{data[0], data[1], data[2], data[3]};
            i += 1;
            data += 4;
        }
    }

    static float4 make_float4(float i, float j, float k, float l) {
        float4 *f = new float4(i, j, k, l);
        return std::move(*f);
    }
};

struct dim3 {
    unsigned int x, y, z;
};

float fmaf(float x, float y, float z) { return x * y + z; }

// Kernel for shifting all matrix elements by a value, A = a + A
void addScalar(float4 *data, float scalar, size_t start, size_t end) {

  for (size_t i = start; i < end; i++) {
    float4 val = data[i];
    val.x += scalar;
    val.y += scalar;
    val.z += scalar;
    val.w += scalar;
    data[i] = val;
  }

}

void launchAddScalar(float *d_data, float scalar, size_t rows, size_t cols, uint8_t thread_id) {
  
  size_t n = rows * cols;

  float4 *data_vec;
  float4::create_float4_vec(data_vec, d_data);

  // Calculate grid size based on vectorized element count
  size_t n_vec = n / 4; 

  size_t start = n_vec / thread_id;
  size_t end   = n_vec / (thread_id + 1);

  addScalar(data_vec, scalar, start, end);

  if (thread_id == thread_count) {
    // Handle the tail elements (0 to 3 elements)
    size_t tail_start = n_vec * 4;
    for (size_t i = tail_start; i < n; i++) {
      d_data[i] += scalar;
    }
  }

}

void launchSubtractScalar(float *d_data, float scalar, size_t rows, size_t cols, uint8_t thread_id) {
  
  size_t n = rows * cols;
  scalar = (-1.0f)* scalar;

  float4 *data_vec;
  float4::create_float4_vec(data_vec, d_data);

  // Calculate grid size based on vectorized element count
  size_t n_vec = n / 4; 

  size_t start = n_vec / thread_id;
  size_t end   = n_vec / (thread_id + 1);

  addScalar(data_vec, scalar, start, end);

  if (thread_id == thread_count) {
    // Handle the tail elements (0 to 3 elements)
    size_t tail_start = n_vec * 4;
    for (size_t i = tail_start; i < n; i++) {
      d_data[i] += scalar;
    }
  }

}

// Vectorized kernel for matrix scaling
void multiplyScalar(float4 *data, float factor, size_t start, size_t end) {
//   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
//   size_t stride = blockDim.x * gridDim.x;

  

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = start; i < end; i++) {
    float4 val = data[i];
    val.x *= factor;
    val.y *= factor;
    val.z *= factor;
    val.w *= factor;
    data[i] = val;
  }

  
}

void launchMultiplyScalar(float *d_data, float factor, size_t rows, size_t cols, uint8_t thread_id) {
  
  size_t n = rows * cols;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  float4 *data_vec;
  float4::create_float4_vec(data_vec, d_data);

  size_t start = n_vec / thread_id;
  size_t end   = n_vec / (thread_id + 1);

  multiplyScalar(data_vec, factor, start, end);
  
  if (thread_id == thread_count) {
    // Tail processing for remaining elements (0 to 3 floats)
    size_t tail_start = n_vec * 4;
    for (size_t i = tail_start; i < n; i++) {
      d_data[i] *= factor;
    }
  }
}

// Vectorized kernel for fused multiply-add (y = alpha * x + beta)
void fusedSclarMultiplyAndAdd(float4 *data, float alpha, float beta, size_t start, size_t end) {

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = start; i < end; i++) {
    float4 val = data[i];
    
    // Hardware Fused Multiply-Add (FMA)
    val.x = fmaf(val.x, alpha, beta);
    val.y = fmaf(val.y, alpha, beta);
    val.z = fmaf(val.z, alpha, beta);
    val.w = fmaf(val.w, alpha, beta);
    
    data[i] = val;
  }

  
}

void launchFusedScalarMultiplyAndAdd(float *d_data, float alpha, float beta, size_t rows, size_t cols, uint8_t thread_id) {
  size_t n = rows * cols;
  int blockSize = 256;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  float4 *data_vec;
  float4::create_float4_vec(data_vec, d_data);

  fusedSclarMultiplyAndAdd(data_vec, alpha, beta, start, end);
  if (thread_id == thread_count) {
    // Tail processing
    size_t tail_start = n_vec * 4;
    for (size_t i = tail_start; i < n; i++) {
      d_data[i] = fmaf(d_data[i], alpha, beta);
    }
  }
}

// Vectorized kernel for matrix-matrix addition (C = A + B)
void matrixAddVectorized(const float4 *A, const float4 *B, float4 *C, size_t start, size_t end) {

  

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = start; i < end; i++) {
    float4 a = A[i];
    float4 b = B[i];
    
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    
    // Single 128-bit store
    C[i] = c;
  }

  
}

void launchMatrixAdd(const float *d_A, const float *d_B, float *d_C, size_t rows, size_t cols, uint8_t thread_id) {
  
  size_t n = rows * cols;

  size_t n_vec = n / 4;
  
  float4 *A_vec;
  float4::create_float4_vec(A_vec, d_A);
  float4 *B_vec;
  float4::create_float4_vec(B_vec, d_B);
  float4 *C_vec;
  float4::create_float4_vec(C_vec, d_C);

  size_t start = n_vec / thread_id;
  size_t end   = n_vec / (thread_id + 1);

  matrixAddVectorized(A_vec, B_vec, C_vec, start, end);

  if (thread_id == thread_count)
    // Tail processing for arrays not perfectly divisible by 4
    size_t tail_start = n_vec * 4;
    for (size_t i = tail_start; i < n; i++) {
      d_C[i] = d_A[i] + d_B[i];
    }
}

// Vectorized kernel for in-place matrix addition (A += B)
void inPlaceMatrixAddVectorized(float4* __restrict__ A, const float4* __restrict__ B, size_t start, size_t end) {
//   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
//   size_t stride = blockDim.x * gridDim.x;

  

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = start; i < end; i++) {
    // Load 128-bits from A and B
    float4 a = A[i];
    float4 b = B[i];
    
    // Perform vector addition locally in registers
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    
    // Store 128-bits back into A
    A[i] = a;
  }

  
}

void launchInPlaceMatrixAdd(float *d_A, const float *d_B, size_t rows, size_t cols, uint8_t thread_id) {
  
  size_t n = rows * cols;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  
  // Cast pointers to float4 for 128-bit memory transactions.
  float4 *A_vec;
  float4::create_float4_vec(A_vec, d_A);
  float4 *B_vec;
  float4::create_float4_vec(B_vec, d_B);

  size_t start = n_vec / thread_id;
  size_t end   = n_vec / (thread_id + 1);

  inPlaceMatrixAddVectorized(A_vec, B_vec, start, end);

  if (thread_id == thread_count) {
    // Tail processing for arrays not perfectly divisible by 4
    size_t tail_start = n_vec * 4;
    for (size_t i = tail_start; i < n; i++) {
      d_A[i] += d_B[i];
    }
  }
}

// Vectorized kernel for matrix-matrix subtraction (C = A - B)
void matrixSubVectorized(const float4* __restrict__ A, 
                                          const float4* __restrict__ B, 
                                          float4* __restrict__ C, 
                                          size_t start, size_t end) {

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = start; i < end; i++) {
    float4 a = A[i];
    float4 b = B[i];
    
    float4 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;
    c.w = a.w - b.w;
    
    // Single 128-bit store
    C[i] = c;
  }


}

void launchMatrixSub(const float *d_A, const float *d_B, float *d_C, size_t rows, size_t cols, uint8_t thread_id) {
  size_t n = rows * cols;
  // Vectorized processing setup
  size_t n_vec = n / 4;
  
  // Cast pointers to float4 for 128-bit memory transactions.
  // A, B, and C must be 16-byte aligned.
  float4 *A_vec;
  float4::create_float4_vec(A_vec, d_A);
  float4 *B_vec;
  float4::create_float4_vec(B_vec, d_B);
  float4 *C_vec;
  float4::create_float4_vec(C_vec, d_C);

  size_t start = n_vec / thread_id;
  size_t end   = n_vec / (thread_id + 1);

  matrixSubVectorized(A_vec, B_vec, C_vec, start, end);

  if (thread_id == thread_count) {
      // Tail processing for arrays not perfectly divisible by 4
    size_t tail_start = n_vec * 4;
    for (size_t i = tail_start; i < n; i++) {
      d_C[i] = d_A[i] - d_B[i];
    }
  }
}

// Vectorized kernel for in-place matrix subtraction (A -= B)
void inPlaceMatrixSubVectorized(float4* __restrict__ A, 
                                                 const float4* __restrict__ B, 
                                                 size_t start, size_t end) {

  
  // Main grid-stride loop over 128-bit chunks
  for (size_t i = start; i < end; i++) {
    float4 a = A[i];
    float4 b = B[i];
    
    // Perform vector subtraction locally in registers
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    
    // Store the 128-bit result back into A
    A[i] = a;
  }

  
}

void launchInPlaceMatrixSub(float *d_A, const float *d_B, size_t rows, size_t cols, uint8_t thread_id) {
  
  size_t n = rows * cols;
  // Vectorized processing setup
  size_t n_vec = n / 4;
  
  // Cast pointers to float4 for 128-bit memory transactions.
  // A is mutable (read/write), B is read-only.
  float4 *A_vec;
  float4::create_float4_vec(A_vec, d_A);
  float4 *B_vec;
  float4::create_float4_vec(B_vec, d_B);


  size_t start = n_vec / thread_id;
  size_t end   = n_vec / (thread_id + 1);

  inPlaceMatrixSubVectorized(d_A, d_B, start, end);
  if (thread_id == thread_count) {
    // Tail processing for arrays not perfectly divisible by 4
    size_t tail_start = n_vec * 4;
    for (size_t i = tail_start; i < n; i++) {
      A[i] -= B[i];
    }
  }
}

/// TODO: convert this to CPU (I don't know what's going on here)
// GEMM using Shared Memory Tiling
void sgemmSharedMemoryTiled(const float* __restrict__ A, 
                                             const float* __restrict__ B, 
                                             float* __restrict__ C, 
                                             int M, int N, int K) {
  // Allocate static shared memory for the tiles of A and B
// float As[TILE_SIZE][TILE_SIZE];
// float Bs[TILE_SIZE][TILE_SIZE];

//   // Calculate global row and column indices for the C matrix
//   int row = blockIdx.y * blockDim.y + threadIdx.y;
//   int col = blockIdx.x * blockDim.x + threadIdx.x;

//   // Local register to accumulate the dot product
//   float sum = 0.0f;

//   // Loop over the K dimension in steps of TILE_SIZE
//   int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
//   for (int t = 0; t < numTiles; ++t) {
    
//     // Load data from global to shared memory.
//     // Include bounds checking for matrices whose dimensions are not multiples of TILE_SIZE.
//     if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
//       As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
//     } else {
//       As[threadIdx.y][threadIdx.x] = 0.0f; // Zero-pad out-of-bounds
//     }

//     if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
//       Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
//     } else {
//       Bs[threadIdx.y][threadIdx.x] = 0.0f; // Zero-pad out-of-bounds
//     }

//     // Synchronize to ensure all threads in the block have finished loading the tile
    

//     // Compute the partial dot product for the current tile
//     #pragma unroll
//     for (int i = 0; i < TILE_SIZE; ++i) {
//       sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
//     }

//     // Synchronize again to ensure all threads are done computing before 
//     // the next iteration overwrites the shared memory tiles
    
//   }

//   // Write the final accumulated sum to global memory
//   if (row < M && col < N) {
//     C[row * N + col] = sum;
//   }
}

/// TODO: convert this to CPU (I don't know what's going on here)
void launchSgemm(const float *d_A, const float *d_B, float *d_C, int M, int N, int K) {
  // Use a 2D block topology mapping directly to the tile size
//   dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  
//   // Calculate grid dimensions to cover the entire M x N output matrix
//   dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
//                  (M + TILE_SIZE - 1) / TILE_SIZE);

//   sgemmSharedMemoryTiled(d_A, d_B, d_C, M, N, K);
}

// Fused GEMM + Matrix Addition = (A x B) + D
void sgemmAddFused(const float* __restrict__ A, 
                                    const float* __restrict__ B, 
                                    const float* __restrict__ D,
                                    float* __restrict__ C, 
                                    int M, int N, int K) {
float As[TILE_SIZE][TILE_SIZE];
float Bs[TILE_SIZE][TILE_SIZE];

//   int row = blockIdx.y * blockDim.y + threadIdx.y;
//   int col = blockIdx.x * blockDim.x + threadIdx.x;

//   for (int row = 0; row < ; row++) {
//     for (int col = 0; col < ; col++) {
//         float sum = 0.0f;

//         int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
//         for (int t = 0; t < numTiles; ++t) {
//             // Collaborative load of A tile into shared memory
//             if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
//             As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
//             } else {
//             As[threadIdx.y][threadIdx.x] = 0.0f; 
//             }

//             // Collaborative load of B tile into shared memory
//             if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
//             Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
//             } else {
//             Bs[threadIdx.y][threadIdx.x] = 0.0f; 
//             }

            

//             #pragma unroll
//             for (int i = 0; i < TILE_SIZE; ++i) {
//             sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
//             }
            
//         }

//         // Add the D matrix
//         if (row < M && col < N) {
//             int idx = row * N + col;

//             // Read D once, add to the register, then store to C
//             C[idx] = sum + D[idx];
//         }
//     }
//   }
}

void launchSgemmAddFused(const float *d_A, const float *d_B, const float *d_D, float *d_C, 
                         int M, int N, int K) {
                         
//   // Define a 2D block matching the tile dimensions.
//   dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  
//   // Calculate grid dimensions to ensure we launch enough blocks to cover the output matrix C.
//   dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
//                  (M + TILE_SIZE - 1) / TILE_SIZE);

//   sgemmAddFused(d_A, d_B, d_D, d_C, M, N, K);
  
}

// Fused Matrix Addition and Scaling: C = (alpha * A) + (beta * B)
void matrixScaleAddVectorized(const float* __restrict__ A, 
                                               const float* __restrict__ B, 
                                               float* __restrict__ C, 
                                               float alpha, float beta, 
                                               size_t start, size_t end) {

  

  for (size_t i = start; i < end; i++) {
    float4 a = A[i];
    float4 b = B[i];
    float4 c;
    
    // Using FMA (Fused Multiply-Add) to execute
    c.x = fmaf(a.x, alpha, b.x * beta);
    c.y = fmaf(a.y, alpha, b.y * beta);
    c.z = fmaf(a.z, alpha, b.z * beta);
    c.w = fmaf(a.w, alpha, b.w * beta);
    
    C[i] = c;
  }

  
}

void launchMatrixScaleAdd(const float *d_A, const float *d_B, float *d_C, 
                          float alpha, float beta, size_t rows, size_t cols, uint8_t thread_id) {
  
  size_t n = rows * cols;
  
  size_t n_vec = n / 4;
  
  float4 *A_vec;
  float4::create_float4_vec(A_vec, d_A);
  float4 *B_vec;
  float4::create_float4_vec(B_vec, d_B);
  float4 *C_vec;
  float4::create_float4_vec(C_vec, d_C);

  size_t start = n_vec / thread_id;
  size_t end   = n_vec / (thread_id + 1);

  // Launch the kernel
  matrixScaleAddVectorized(d_A, d_B, d_C, alpha, beta, start, end);

  if (thread_id == thread_count) {
    size_t tail_start = n_vec * 4;
    for (size_t i = tail_start; i < n; i++) {
      C[i] = fmaf(A[i], alpha, B[i] * beta);
    }
  }
  
}

// Optimized GEMM with In-Place Accumulation (C = alpha * (A * B) + beta * C)
void sgemmInPlaceAccumulate(const float* __restrict__ A, 
                                             const float* __restrict__ B, 
                                             float* __restrict__ C, 
                                             int M, int N, int K,
                                             float alpha, float beta) {
  
// float As[TILE_SIZE][TILE_SIZE];
// float Bs[TILE_SIZE][TILE_SIZE];

//   int row = blockIdx.y * blockDim.y + threadIdx.y;
//   int col = blockIdx.x * blockDim.x + threadIdx.x;

//   float sum = 0.0f;

//   int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
//   for (int t = 0; t < numTiles; ++t) {
    
//     // Load tiles into shared memory
//     if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
//       As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
//     } else {
//       As[threadIdx.y][threadIdx.x] = 0.0f;
//     }

//     if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
//       Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
//     } else {
//       Bs[threadIdx.y][threadIdx.x] = 0.0f;
//     }

    

//     // Compute partial dot product
//     #pragma unroll
//     for (int i = 0; i < TILE_SIZE; ++i) {
//       sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
//     }

    
//   }

//   // Write the accumulated sum back to global memory IN PLACE
//   if (row < M && col < N) {
//     int idx = row * N + col;
//     // Scale the dot product by alpha, add it to the scaled existing C value
//     C[idx] = alpha * sum + beta * C[idx];
//   }
}

void launchSgemmAccumulate(const float *d_A, const float *d_B, float *d_C, 
                           int M, int N, int K, float alpha = 1.0f, float beta = 1.0f) {
//   dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
//   dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
//                  (M + TILE_SIZE - 1) / TILE_SIZE);

//   sgemmInPlaceAccumulate(d_A, d_B, d_C, M, N, K, alpha, beta);
}

// Vectorized kernel for elementwies matrix-matrix multiplication (C = A . B)
void matrixElementwiseMultVectorized(const float4* A, const float4* B, float4* C, size_t start, size_t end) {

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = start; i < end; i++) {
    float4 a = A[i];
    float4 b = B[i];
    
    float4 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.z = a.z * b.z;
    c.w = a.w * b.w;
    
    // Single 128-bit store
    C[i] = c;
  }

  
}

void launchElementwiseMatrixMult(const float *d_A, const float *d_B, float *d_C, size_t rows, size_t cols, uint8_t thread_id) {
  size_t n = rows * cols;
  // Vectorized processing setup
  size_t n_vec = n / 4;
  
  // Cast pointers to float4 for 128-bit memory transactions.
  // A, B, and C must all be 16-byte aligned.
  float4 *A_vec;
  float4::create_float4_vec(A_vec, d_A);
  float4 *B_vec;
  float4::create_float4_vec(B_vec, d_B);
  float4 *C_vec;
  float4::create_float4_vec(C_vec, d_C);

  size_t start = n_vec / thread_id;
  size_t end   = n_vec / (thread_id + 1);

  matrixElementwiseMultVectorized(d_A, d_B, d_C, start, end);

  if (thread_id == thread_count) {
    // Tail processing for arrays not perfectly divisible by 4
    size_t tail_start = n_vec * 4;
    for (size_t i = tail_start; i < n; i++) {
      C[i] = A[i] * B[i];
    }
  }
}

// Vectorized kernel for in-place element wise matrix multiplication (A .= B)
void inPlaceMatrixElementwiseMultVectorized(float4* __restrict__ A, const float4* __restrict__ B, size_t start, size_t end) {

  // Main grid-stride loop over 128-bit chunks
  for (size_t i = start; i < end; i++) {
    // Load 128-bits from A and B
    float4 a = A[i];
    float4 b = B[i];
    
    // Perform vector addition locally in registers
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    
    // Store 128-bits back into A
    A[i] = a;
  }

  
}

void launchInPlaceMatrixElementwiseMultVectorized(float *d_A, const float *d_B, size_t rows, size_t cols, uint8_t thread_id) {
  
  size_t n = rows * cols;

  // Vectorized processing setup
  size_t n_vec = n / 4;
  
  // Cast pointers to float4 for 128-bit memory transactions.
  float4 *A_vec;
  float4::create_float4_vec(A_vec, d_A);
  float4 *B_vec;
  float4::create_float4_vec(B_vec, d_B);

  size_t start = n_vec / thread_id;
  size_t end   = n_vec / (thread_id + 1);

  inPlaceMatrixElementwiseMultVectorized(d_A, d_B, start, end);

  if (thread_id == thread_count) {
    // Tail processing for arrays not perfectly divisible by 4
  size_t tail_start = n_vec * 4;
  for (size_t i = tail_start; i < n; i++) {
    A[i] *= B[i];
  }
  }
}