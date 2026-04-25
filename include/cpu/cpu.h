#pragma once

#include <omp.h>
#include <algorithm>
#include <cstring> // For std::memcpy

#include "scheduler.h"
#include "cpu_matrix_ops.h"
#include "DenseMat.h"
#include "state.h"
#include "dag.h" // Ensure dag.h is included for the updated DagNode

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

    for (int i = 0; i < 5; i++) {
      const DenseMat<float>& cpu_mat = initial_state.getMatrix(i);
      std::memcpy(host_mats[i], cpu_mat.data(), bytes);
    } 
  }

  void run(const std::map<uint64_t, DagNode>& dag, const std::vector<std::vector<uint64_t>>& levels) {
    // Explicitly enable nested parallelism
    omp_set_max_active_levels(2);
    int total_hw_threads = omp_get_max_threads();

    // Iterate through each level sequentially
    for (const auto& level : levels) {
      int num_ops = level.size();

      // Dynamic Sub-team Allocation
      int threads_per_op = std::max(1, total_hw_threads / num_ops);

      #pragma omp parallel for num_threads(num_ops) schedule(dynamic)
      for (size_t i = 0; i < level.size(); ++i) {
        uint64_t op_id = level[i];
        const DagNode& node = dag.at(op_id);
        
        // Get the current physical thread ID to access the correct workspace
        int thread_idx = omp_get_thread_num();

        // Set number of parallel threads for individual operation
        omp_set_num_threads(threads_per_op);

        launch(node, thread_idx);
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