#pragma once

#include <algorithm>
#include <atomic>
#include <cstring>
#include <unordered_map>
#include <thread>
#include <barrier>
#include <memory>
#include <cuda_runtime.h>

#include "DenseMat.h"
#include "cpu_matrix_ops.h"
#include "dag.h"
#include "state.h"
#include "workload.h"

class CpuExecutor {
private:
  float** host_mats_;                     // Pinned CPU Host Memory
  float** device_mats_;                   // Dedicated GPU VRAM (for explicit DMA pulls)
  std::vector<float*> thread_workspaces_; // Pre-allocated workspaces per thread for MAT_MULT
  uint64_t rows_, cols_;
  uint64_t num_matrices_;

  std::vector<std::thread> workers;
  std::unique_ptr<std::barrier<>> sync_barrier;
  std::atomic<bool> terminate_pool{false};

  // Shared pointers for the currently executing level
  const std::unordered_map<uint64_t, DagNode>* current_dag = nullptr;
  const std::vector<uint64_t>* current_tasks = nullptr;
  std::atomic<size_t> next_task_idx{0};
  std::atomic<int>* current_op_counter = nullptr;

public:
  CpuExecutor(uint64_t matrix_dim, uint64_t num_matrices, float** h_mats, float** d_mats = nullptr)
      : host_mats_(h_mats), device_mats_(d_mats), rows_(matrix_dim), cols_(matrix_dim), num_matrices_(num_matrices) {

    // Initialize physical threads matching hardware concurrency
    int num_workers = std::max(1u, std::thread::hardware_concurrency());
    thread_workspaces_.resize(num_workers);
    
    for (int i = 0; i < num_workers; ++i) {
      thread_workspaces_[i] = new float[rows_ * cols_]();
    }

    // Initialize barrier for the workers + 1 master thread
    sync_barrier = std::make_unique<std::barrier<>>(num_workers + 1);

    // Pre-spawn the task-level thread pool
    for (int i = 0; i < num_workers; ++i) {
      workers.emplace_back(&CpuExecutor::worker_loop, this, i);
    }
  }

  ~CpuExecutor() {
    // Signal workers to terminate and drop the barrier to wake them up
    terminate_pool.store(true, std::memory_order_relaxed);
    sync_barrier->arrive_and_wait();

    for (auto& t : workers) {
      if (t.joinable())
        t.join();
    }

    for (size_t i = 0; i < thread_workspaces_.size(); ++i) {
      delete[] thread_workspaces_[i];
    }
  }

  void load_state(const State<float>& initial_state) {
    size_t bytes = rows_ * cols_ * sizeof(float);

    for (int i = 0; i < (int)num_matrices_; i++) {
      const DenseMat<float>& cpu_mat = initial_state.getMatrix(i);
      std::memcpy(host_mats_[i], cpu_mat.data(), bytes);
    }
  }

  void run(const std::unordered_map<uint64_t, DagNode>& dag, const std::vector<std::vector<uint64_t>>& levels, std::atomic<int> *op_counter) {
    int total_hw_threads = static_cast<int>(thread_workspaces_.size());

    // Iterate through each level sequentially on the master thread
    for (const auto& level : levels) {
      
      // Filter out GPU tasks
      std::vector<uint64_t> cpu_tasks;
      cpu_tasks.reserve(level.size());

      std::vector<uint64_t> mats_to_pull;
      mats_to_pull.reserve(level.size() * 2);

      for (uint64_t op_id : level) {
        const DagNode& node = dag.at(op_id);
        if (node.target == ExecTarget::CPU) {
          cpu_tasks.push_back(op_id);

          // Check dependencies to see if the GPU was the last writer
          for (int i = 0; i < node.dep_count; ++i) {
            uint64_t dep_id = node.deps[i];
            const DagNode& dep_node = dag.at(dep_id);

            if (dep_node.target == ExecTarget::GPU && device_mats_ != nullptr) {
              mats_to_pull.push_back(dep_node.operation.dest_mat_id_1.value());
            }
          }
        }
      }

      if (cpu_tasks.empty()) continue;

      // Bulk transfer GPU-mutated matrices into CPU Host Memory
      if (!mats_to_pull.empty()) {
        // Remove duplicates so we don't pull the same matrix twice
        std::sort(mats_to_pull.begin(), mats_to_pull.end());
        mats_to_pull.erase(std::unique(mats_to_pull.begin(), mats_to_pull.end()), mats_to_pull.end());

        size_t bytes = rows_ * cols_ * sizeof(float);
        for (uint64_t mat_id : mats_to_pull) {
          cudaMemcpyAsync(host_mats_[mat_id], device_mats_[mat_id], bytes, cudaMemcpyDeviceToHost, 0);
        }
        
        // We must block the master thread until the D2H bulk transfer finishes.
        cudaStreamSynchronize(0); 
      }

      // STRATEGY 1: Task-Level Parallelism
      // We have enough operations to keep the cores busy. 
      // 1 Worker Thread = 1 Operation. Internal math is strictly sequential.
      if ((int)cpu_tasks.size() >= total_hw_threads) {
        
        // Stage the work
        current_dag = &dag;
        current_tasks = &cpu_tasks;
        current_op_counter = op_counter;
        next_task_idx.store(0, std::memory_order_relaxed);

        // Drop barrier to release workers
        sync_barrier->arrive_and_wait();

        // Wait at the barrier until all workers finish the level
        sync_barrier->arrive_and_wait();

        // Bulk-update the op_counter
        if (op_counter) {
          int level_ops = 0;
          for (uint64_t op_id : cpu_tasks) {
            level_ops += dag.at(op_id).original_op_count;
          }
          op_counter->fetch_add(level_ops, std::memory_order_relaxed);
        }
      }
      // STRATEGY 2: Data-Level Parallelism
      // Few operations. Master thread executes them, but passes 'use_parallel = true'
      // to wake up the internal MathThreadPool.
      else {
        for (uint64_t op_id : cpu_tasks) {
          const DagNode& node = dag.at(op_id);
          
          // Execute on master thread, delegating inner math to multiple threads.
          // Note: using workspace 0 since the master thread handles this block.
          launch(node, 0, true); 

          if (op_counter) {
            op_counter->fetch_add(node.original_op_count, std::memory_order_relaxed);
          }
        }
      }
    }
  }

  // Sequential Execution Baseline (CPU)
  void run_sequential(const std::unordered_map<uint64_t, DagNode>& dag, std::atomic<int> *op_counter) {
    size_t bytes = rows_ * cols_ * sizeof(float);

    // Because unordered_map loses generation order, we must manually 
    // sort the operation IDs to ensure proper sequential state machine execution.
    std::vector<uint64_t> sorted_keys;
    sorted_keys.reserve(dag.size());
    for (const auto& pair : dag) {
      sorted_keys.push_back(pair.first);
    }
    std::sort(sorted_keys.begin(), sorted_keys.end());

    // Iterate in ascending op_id order
    for (uint64_t op_id : sorted_keys) {
      const DagNode& node = dag.at(op_id);

      // Skip GPU operations
      if (node.target != ExecTarget::CPU)
        continue;
      
      // Sequential mode also requires state pulls if the last writer was the GPU
      for (int i = 0; i < node.dep_count; ++i) {
        uint64_t dep_id = node.deps[i];
        const DagNode& dep_node = dag.at(dep_id);

        if (dep_node.target == ExecTarget::GPU && device_mats_ != nullptr) {
          uint32_t gpu_mutated_mat = dep_node.operation.dest_mat_id_1.value();
          cudaMemcpy(host_mats_[gpu_mutated_mat],
                     device_mats_[gpu_mutated_mat],
                     bytes,
                     cudaMemcpyDeviceToHost);
        }
      }

      // Execute sequentially on the main thread. Internal math is strictly sequential.
      launch(node, 0, false);

      // Increment by original_op_count
      if (op_counter) {
        op_counter->fetch_add(node.original_op_count,
                              std::memory_order_relaxed);
      }
    }
  }

private:
  // The persistent loop executed by the pre-spawned task threads
  // The persistent loop executed by the pre-spawned task threads
  void worker_loop(int thread_idx) {
    while (true) {
      // Block here until the master thread stages a level
      sync_barrier->arrive_and_wait();

      // Check for signal
      if (terminate_pool.load(std::memory_order_relaxed))
        break;

      // Task grabbing
      while (true) {
        size_t idx = next_task_idx.fetch_add(1, std::memory_order_relaxed);
        
        // If the index exceeds the task list, this level is complete
        if (idx >= current_tasks->size())
          break;

        uint64_t op_id = (*current_tasks)[idx];
        const DagNode& node = current_dag->at(op_id);

        // Execute task. Task workers NEVER use internal data parallelism.
        launch(node, thread_idx, false);
      }

      // Block here to signal to the master thread that this worker is done with the level
      sync_barrier->arrive_and_wait();
    }
  }

  // Modified to pass the use_parallel flag
  void launch(const DagNode& node, int thread_idx, bool use_parallel) {
    float* d_out = host_mats_[node.operation.dest_mat_id_1.value()];

    if (node.has_fused_scalar) {
      fusedScalarMultiplyAndAddCPU(d_out, node.fused_alpha, node.fused_beta, rows_, cols_, use_parallel);
    } else {
      switch (node.operation.type) {
        case OpType::SCALAR_ADD:
          addScalarCPU(d_out, (float)node.operation.scalar_param.value(), rows_, cols_, use_parallel);
          break;

        case OpType::SCALAR_SUB:
          subtractScalarCPU(d_out, (float)node.operation.scalar_param.value(), rows_, cols_, use_parallel);
          break;

        case OpType::SCALAR_MULT:
          multiplyScalarCPU(d_out, (float)node.operation.scalar_param.value(), rows_, cols_, use_parallel);
          break;

        case OpType::MAT_MULT: {
          float* d_mat_B = host_mats_[node.operation.dest_mat_id_2.value()];
          float* temp_result = thread_workspaces_[thread_idx];
          size_t bytes = rows_ * cols_ * sizeof(float);

          // Compute into temp buffer, then synchronous memory copy
          launchSgemmCPU(d_out, d_mat_B, temp_result, rows_, cols_, cols_, use_parallel);
          std::memcpy(d_out, temp_result, bytes);
          break;
        }

        case OpType::NEW_MAT_ADD:
          inPlaceMatrixAddCPU(d_out, node.h_mat_param, rows_, cols_, use_parallel); 
          break; 

        case OpType::NEW_MAT_SUB:
          inPlaceMatrixSubCPU(d_out, node.h_mat_param, rows_, cols_, use_parallel); 
          break;

        case OpType::NEW_MAT_MULT: {
          float* temp_result = thread_workspaces_[thread_idx];
          size_t bytes = rows_ * cols_ * sizeof(float);

          launchSgemmCPU(d_out, node.h_mat_param, temp_result, rows_, cols_, cols_, use_parallel);
          std::memcpy(d_out, temp_result, bytes);
          break;
        }
        case OpType::ELEMAT_MULT:
          inPlaceElementwiseMatrixMultCPU(d_out, node.h_mat_param, rows_, cols_, use_parallel);
          break;

        default:
          break;
      }
    }
  }
};