#pragma once

#include <omp.h>
#include <algorithm>
#include <cstring>
#include <atomic>
#include <vector>
#include <unordered_map>

#include "cpu_matrix_ops.h"
#include "DenseMat.h"
#include "state.h"
#include "dag.h"
#include "workload.h"

class CpuExecutor {
private:
  float *host_mats[5];                    // Pointers to the 5 state matrices in system RAM
  std::vector<float *> thread_workspaces; // Pre-allocated workspaces per thread for MAT_MULT
  uint64_t rows, cols;

public:
  CpuExecutor(uint64_t r, uint64_t c) : rows(r), cols(c) {
    // Allocate State Matrices in standard RAM
    for (int i = 0; i < 5; i++) {
      host_mats[i] = new float[rows * cols]();
    } 

    int max_threads = omp_get_max_threads();
    thread_workspaces.resize(max_threads);
    
    for (int i = 0; i < max_threads; ++i) {
      thread_workspaces[i] = new float[rows * cols]();
    }
  }

  ~CpuExecutor() {
    for (int i = 0; i < 5; ++i) {
      delete[] host_mats[i];
    }
    for (size_t i = 0; i < thread_workspaces.size(); ++i) {
      delete[] thread_workspaces[i];
    }
  }

  void load_state(const State<float>& initial_state) {
    size_t bytes = rows * cols * sizeof(float); 

    for (int i = 0; i < kNumMatrices; i++) {
      const DenseMat<float>& cpu_mat = initial_state.getMatrix(i);
      std::memcpy(host_mats[i], cpu_mat.data(), bytes);
    } 
  }

  void run(const std::unordered_map<uint64_t, DagNode>& dag, const std::vector<std::vector<uint64_t>>& levels, std::atomic<int> *op_counter) {
    // Disable nested parallelism to prevent thread oversubscription and memory safety issues.
    omp_set_max_active_levels(1);

    // Iterate through each level sequentially
    for (const auto& level : levels) {
      
      // Filter out GPU tasks to avoid indexing errors
      std::vector<uint64_t> cpu_tasks;
      for (uint64_t op_id : level) {
        if (dag.at(op_id).target == ExecTarget::CPU) {
          cpu_tasks.push_back(op_id);
        }
      }

      if (cpu_tasks.empty())
        continue;

      // Distribute CPU operations across the available physical cores
      #pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < cpu_tasks.size(); ++i) {
        uint64_t op_id = cpu_tasks[i];
        const DagNode& node = dag.at(op_id);
        
        // Safe physical thread ID to access the correct workspace
        int thread_idx = omp_get_thread_num();

        launch(node, thread_idx);

        // Use original_op_count to properly credit fused operations
        if (op_counter) {
          op_counter->fetch_add(node.original_op_count, std::memory_order_relaxed);
        }
      }
    } 
  }

  // Sequential Execution Baseline (CPU)
  void run_sequential(const std::unordered_map<uint64_t, DagNode>& dag, std::atomic<int> *op_counter) {
    // Guarantee that the single operation gets 100% of the physical cores 
    // to execute its internal SIMD loops.
    omp_set_num_threads(omp_get_max_threads());

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
      
      // Execute sequentially on the main thread, using workspace 0
      launch(node, 0);

      // Increment by original_op_count
      if (op_counter) {
        op_counter->fetch_add(node.original_op_count, std::memory_order_relaxed);
      }
    }
  }

private:
  void launch(const DagNode& node, int thread_idx) {
    float* d_out = host_mats[node.operation.dest_mat_id_1.value()];

    if (node.has_fused_scalar) {
      fusedScalarMultiplyAndAddCPU(d_out, node.fused_alpha, node.fused_beta, rows, cols);
    } else {
      switch (node.operation.type) {
        case OpType::SCALAR_ADD:
          addScalarCPU(d_out, (float)node.operation.scalar_param.value(), rows, cols);
          break;

        case OpType::SCALAR_SUB:
          subtractScalarCPU(d_out, (float)node.operation.scalar_param.value(), rows, cols);
          break;

        case OpType::SCALAR_MULT:
          multiplyScalarCPU(d_out, (float)node.operation.scalar_param.value(), rows, cols);
          break;

        case OpType::MAT_MULT: 
        {
          float* d_mat_B = host_mats[node.operation.dest_mat_id_2.value()];
          float* temp_result = thread_workspaces[thread_idx]; 
          size_t bytes = rows * cols * sizeof(float);

          // Compute into temp buffer, then synchronous memory copy
          launchSgemmCPU(d_out, d_mat_B, temp_result, rows, cols, cols);
          std::memcpy(d_out, temp_result, bytes);
          break;
        }

        // FIXED: Replaced node.d_mat_param with node.h_mat_param
        case OpType::NEW_MAT_ADD:
          inPlaceMatrixAddCPU(d_out, node.h_mat_param, rows, cols); 
          break; 

        case OpType::NEW_MAT_SUB:
          inPlaceMatrixSubCPU(d_out, node.h_mat_param, rows, cols); 
          break;

        case OpType::NEW_MAT_MULT:
        {
          float* temp_result = thread_workspaces[thread_idx]; 
          size_t bytes = rows * cols * sizeof(float);

          launchSgemmCPU(d_out, node.h_mat_param, temp_result, rows, cols, cols);
          std::memcpy(d_out, temp_result, bytes);
          break;
        }
        case OpType::ELEMAT_MULT:
          inPlaceElementwiseMatrixMultCPU(d_out, node.h_mat_param, rows, cols);
          break;

        default:
          break;
      } 
    } 
  } 
};