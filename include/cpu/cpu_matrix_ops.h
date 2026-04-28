#pragma once

#include <iostream>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>

class MathThreadPool {
private:
  int num_workers;
  std::vector<std::thread> workers;
  std::atomic<bool> terminate_{false};
  std::atomic<int> active_tasks_{0};
  
  std::mutex mtx_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  
  std::function<void(size_t, size_t)> current_task_;
  size_t total_work_ = 0;
  size_t chunk_size_ = 0;
  int task_generation_ = 0; // FIXED: Prevents workers from double-dipping tasks

  MathThreadPool() {
    num_workers = std::thread::hardware_concurrency() - 1;
    if (num_workers <= 0) num_workers = 1;
    
    for (int i = 0; i < num_workers; ++i) {
      workers.emplace_back([this, i]() {
        int local_gen = 0;
        while (true) {
          std::unique_lock<std::mutex> lock(mtx_);
          
          // Wait for a NEW generation of work, or a termination signal
          cv_start_.wait(lock, [this, local_gen] { 
              return task_generation_ != local_gen || terminate_; 
          });
          
          if (terminate_) return;
          
          local_gen = task_generation_;
          auto task = current_task_;
          size_t start = i * chunk_size_;
          size_t end = std::min(start + chunk_size_, total_work_);
          lock.unlock(); 
          
          if (start < total_work_) {
            task(start, end);
          }
          
          lock.lock();
          if (--active_tasks_ == 0) {
            cv_done_.notify_one(); 
          }
        }
      });
    }
  }

public:
  static MathThreadPool& get() {
    static MathThreadPool instance;
    return instance;
  }

  ~MathThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      terminate_ = true;
    }
    cv_start_.notify_all();
    for (auto& t : workers) {
      if (t.joinable()) t.join();
    }
  }

  template<typename F>
  void parallel_for(size_t total, F&& func) {
    if (total == 0) return;

    std::unique_lock<std::mutex> lock(mtx_);
    total_work_ = total;
    chunk_size_ = (total + num_workers) / (num_workers + 1);
    current_task_ = func;
    active_tasks_ = num_workers;
    task_generation_++; // Increment generation to wake workers safely
    
    cv_start_.notify_all();
    lock.unlock();

    // The calling thread processes the final chunk
    size_t start = num_workers * chunk_size_;
    if (start < total) {
      func(start, total);
    }

    lock.lock();
    cv_done_.wait(lock, [this] { return active_tasks_ == 0; });
    current_task_ = nullptr;
  }
};

constexpr size_t THREAD_THRESHOLD = 16384;

void addScalarCPU(float* __restrict__ data, float scalar, size_t rows, size_t cols, bool use_parallel = false) {
  size_t n = rows * cols;
  auto compute = [&](size_t start, size_t end) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) data[i] += scalar;
  };
  if (use_parallel && n >= THREAD_THRESHOLD)
    MathThreadPool::get().parallel_for(n, compute);
  else
    compute(0, n);
}

void subtractScalarCPU(float* __restrict__ data, float scalar, size_t rows, size_t cols, bool use_parallel = false) {
  size_t n = rows * cols;
  auto compute = [&](size_t start, size_t end) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) data[i] -= scalar;
  };
  if (use_parallel && n >= THREAD_THRESHOLD)
    MathThreadPool::get().parallel_for(n, compute);
  else
    compute(0, n);
}

void multiplyScalarCPU(float* __restrict__ data, float scalar, size_t rows, size_t cols, bool use_parallel = false) {
  size_t n = rows * cols;
  auto compute = [&](size_t start, size_t end) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) data[i] *= scalar;
  };
  if (use_parallel && n >= THREAD_THRESHOLD) 
    MathThreadPool::get().parallel_for(n, compute);
  else
    compute(0, n);
}

void fusedScalarMultiplyAndAddCPU(float* __restrict__ data, float alpha, float beta, size_t rows, size_t cols, bool use_parallel = false) {
  size_t n = rows * cols;
  auto compute = [&](size_t start, size_t end) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) data[i] = std::fma(data[i], alpha, beta);
  };
  if (use_parallel && n >= THREAD_THRESHOLD)
    MathThreadPool::get().parallel_for(n, compute);
  else
    compute(0, n);
}

// C = A + B
void matrixAddCPU(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, size_t rows, size_t cols, bool use_parallel = false) {
  size_t n = rows * cols;
  auto compute = [&](size_t start, size_t end) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) C[i] = A[i] + B[i];
  };
  if (use_parallel && n >= THREAD_THRESHOLD)
    MathThreadPool::get().parallel_for(n, compute);
  else
    compute(0, n);
}

// C = A - B
void matrixSubCPU(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, size_t rows, size_t cols, bool use_parallel = false) {
  size_t n = rows * cols;
  auto compute = [&](size_t start, size_t end) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) C[i] = A[i] - B[i];
  };
  if (use_parallel && n >= THREAD_THRESHOLD)
    MathThreadPool::get().parallel_for(n, compute);
  else
    compute(0, n);
}

// A += B
void inPlaceMatrixAddCPU(float* __restrict__ A, const float* __restrict__ B, size_t rows, size_t cols, bool use_parallel = false) {
  size_t n = rows * cols;
  auto compute = [&](size_t start, size_t end) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) A[i] += B[i];
  };
  if (use_parallel && n >= THREAD_THRESHOLD)
    MathThreadPool::get().parallel_for(n, compute);
  else
    compute(0, n);
}

// A -= B
void inPlaceMatrixSubCPU(float* __restrict__ A, const float* __restrict__ B, size_t rows, size_t cols, bool use_parallel = false) {
  size_t n = rows * cols;
  auto compute = [&](size_t start, size_t end) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) A[i] -= B[i];
  };
  if (use_parallel && n >= THREAD_THRESHOLD)
    MathThreadPool::get().parallel_for(n, compute);
  else
    compute(0, n);
}

// Optimized CPU Matrix Multiplication (C = A * B)
void launchSgemmCPU(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K, bool use_parallel = false) {
  
  auto init_C = [&](size_t start_i, size_t end_i) {
    for (size_t i = start_i; i < end_i; ++i) {
      #pragma omp simd
      for (int j = 0; j < N; ++j) {
        C[i * N + j] = 0.0f;
      }
    }
  };

  auto compute_sgemm = [&](size_t start_i, size_t end_i) {
    for (size_t i = start_i; i < end_i; ++i) {
      for (int k = 0; k < K; ++k) {
        float a_ik = A[i * K + k]; 
        #pragma omp simd
        for (int j = 0; j < N; ++j) {
          C[i * N + j] += a_ik * B[k * N + j];
        }
      }
    }
  };

  if (use_parallel && M >= 64) {
    MathThreadPool::get().parallel_for(M, init_C);
    MathThreadPool::get().parallel_for(M, compute_sgemm);
  } else {
    init_C(0, M);
    compute_sgemm(0, M);
  }
}

// C = A * B + D
void sgemmAddFusedCPU(const float* __restrict__ A, const float* __restrict__ B, const float* __restrict__ D, float* __restrict__ C, int M, int N, int K, bool use_parallel = false) {
  
  auto init_C = [&](size_t start_i, size_t end_i) {
    for (size_t i = start_i; i < end_i; ++i) {
      #pragma omp simd
      for (int j = 0; j < N; ++j) {
        C[i * N + j] = D[i * N + j];
      }
    }
  };

  auto compute_sgemm = [&](size_t start_i, size_t end_i) {
    for (size_t i = start_i; i < end_i; ++i) {
      for (int k = 0; k < K; ++k) {
        float a_ik = A[i * K + k]; 
        #pragma omp simd
        for (int j = 0; j < N; ++j) {
          C[i * N + j] += a_ik * B[k * N + j];
        }
      }
    }
  };

  if (use_parallel && M >= 64) {
    MathThreadPool::get().parallel_for(M, init_C);
    MathThreadPool::get().parallel_for(M, compute_sgemm);
  } else {
    init_C(0, M);
    compute_sgemm(0, M);
  }
}

// C = alpha * A + beta * B
void matrixScaleAddCPU(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, float alpha, float beta, size_t rows, size_t cols, bool use_parallel = false) {
  size_t n = rows * cols;
  auto compute = [&](size_t start, size_t end) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) {
      C[i] = std::fma(A[i], alpha, B[i] * beta);
    }
  };
  if (use_parallel && n >= THREAD_THRESHOLD) 
    MathThreadPool::get().parallel_for(n, compute); 
  else
    compute(0, n);
}

// C = alpha * (A * B) + beta * C
void sgemmInPlaceAccumulateCPU(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K, float alpha, float beta, bool use_parallel = false) {
  
  auto scale_C = [&](size_t start_i, size_t end_i) {
    for (size_t i = start_i; i < end_i; ++i) {
      #pragma omp simd
      for (int j = 0; j < N; ++j) {
        if (beta == 0.0f) C[i * N + j] = 0.0f;
        else C[i * N + j] *= beta;
      }
    }
  };

  auto compute_sgemm = [&](size_t start_i, size_t end_i) {
    for (size_t i = start_i; i < end_i; ++i) {
      for (int k = 0; k < K; ++k) {
        float a_ik_alpha = A[i * K + k] * alpha; 
        #pragma omp simd
        for (int j = 0; j < N; ++j) {
            C[i * N + j] += a_ik_alpha * B[k * N + j];
        }
      }
    }
  };

  if (use_parallel && M >= 64) {
    MathThreadPool::get().parallel_for(M, scale_C);
    MathThreadPool::get().parallel_for(M, compute_sgemm);
  } else {
    scale_C(0, M);
    compute_sgemm(0, M);
  }
}

// C = A .* B
void elementwiseMatrixMultCPU(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, size_t rows, size_t cols, bool use_parallel = false) {
  size_t n = rows * cols;
  auto compute = [&](size_t start, size_t end) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) C[i] = A[i] * B[i];
  };
  if (use_parallel && n >= THREAD_THRESHOLD)
    MathThreadPool::get().parallel_for(n, compute);
  else
    compute(0, n);
}

// A .*= B
void inPlaceElementwiseMatrixMultCPU(float* __restrict__ A, const float* __restrict__ B, size_t rows, size_t cols, bool use_parallel = false) {
  size_t n = rows * cols;
  auto compute = [&](size_t start, size_t end) {
    #pragma omp simd
    for (size_t i = start; i < end; ++i) A[i] *= B[i];
  };
  if (use_parallel && n >= THREAD_THRESHOLD)
    MathThreadPool::get().parallel_for(n, compute);
  else
    compute(0, n);
}