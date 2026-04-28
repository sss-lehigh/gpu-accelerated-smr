#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <vector>

#include "DenseMat.h"
#include "dag.h"
#include "kernels/common.cuh"
#include "kernels/matrix_ops.h"
#include "state.h"
#include "workload.h"

class GpuExecutor {
 private:
  float** device_mats_;       // Ptrs to state mats
  float** stream_workspace_;  // Pre allocated workspace for MAT_MULT
  cudaStream_t* streams_;     // Concurrent hardware queues
  uint64_t rows_, cols_;
  uint64_t num_matrices_;

  // Map to store CUDA events for dependency tracking
  std::unordered_map<uint64_t, cudaEvent_t> node_events;

  // Map to store pre-allocated VRAM pointers for node parameters
  std::unordered_map<uint64_t, float*> d_op_params;

 public:
  GpuExecutor(uint64_t matrix_dim, uint64_t num_matrices)
      : rows_(matrix_dim), cols_(matrix_dim), num_matrices_(num_matrices) {
    device_mats_ = new float*[num_matrices_];
    for (int i = 0; i < (int)num_matrices_; i++) {
      // allocate memory for state mats
      CUDA_CHECK(cudaMalloc(&device_mats_[i], rows_ * cols_ * sizeof(float)));
    }  // end for

    // Initialize streams and workspaces
    streams_ = new cudaStream_t[8];
    stream_workspace_ = new float*[8];
    for (int i = 0; i < 8; ++i) {
      CUDA_CHECK(cudaStreamCreate(&streams_[i]));
      CUDA_CHECK(
          cudaMalloc(&stream_workspace_[i], rows_ * cols_ * sizeof(float)));
    }
  }  // end constructor

  ~GpuExecutor() {
    for (int i = 0; i < (int)num_matrices_; ++i) {
      cudaFree(device_mats_[i]);
    }

    for (int i = 0; i < 8; ++i) {
      cudaFree(stream_workspace_[i]);
      cudaStreamDestroy(streams_[i]);
    }

    for (auto& pair : node_events) {
      cudaEventDestroy(pair.second);
    }

    // Clean up the parameter matrices stored in VRAM
    for (auto& pair : d_op_params) {
      cudaFree(pair.second);
    }
  }  // end destructor

  void load_state(const State<float>& initial_state) {
    size_t bytes = rows_ * cols_ * sizeof(float);

    for (int i = 0; i < (int)num_matrices_; i++) {
      const DenseMat<float>& cpu_mat = initial_state.getMatrix(i);
      CUDA_CHECK(cudaMemcpy(device_mats_[i], cpu_mat.data(), bytes,
                            cudaMemcpyHostToDevice));
    }  // end for

    CUDA_CHECK(cudaDeviceSynchronize());
  }  // end load state

  // Preparation Phase
  void prepare_dag(const std::unordered_map<uint64_t, DagNode>& dag) {
    for (const auto& [op_id, node] : dag) {
      // Only allocate VRAM and transfer over PCIe if the GPU is actually going
      // to execute this node.
      if (node.target != ExecTarget::GPU) continue;

      if (node.h_mat_param != nullptr) {
        float* d_ptr;
        size_t bytes = node.rows * node.cols * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
        CUDA_CHECK(
            cudaMemcpy(d_ptr, node.h_mat_param, bytes, cudaMemcpyHostToDevice));

        d_op_params[node.operation.id] = d_ptr;
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void run(const std::unordered_map<uint64_t, DagNode>& dag,
           const std::vector<std::vector<uint64_t>>& levels,
           std::atomic<int>* op_counter) {
    for (const auto& pair : dag) {
      // Only create events for nodes the GPU will process or wait on
      if (pair.second.target == ExecTarget::GPU) {
        cudaEventCreateWithFlags(&node_events[pair.first],
                                 cudaEventDisableTiming);
      }
    }

    int s_idx = 0;

    for (const auto& level : levels) {
      for (uint64_t op_id : level) {
        const DagNode& node = dag.at(op_id);

        // Skip CPU operations
        if (node.target != ExecTarget::GPU) continue;

        int current_stream_idx = s_idx % 8;
        cudaStream_t stream = streams_[current_stream_idx];

        // Replaced std::set range loop with array counter loop
        for (int i = 0; i < node.dep_count; ++i) {
          uint64_t dep_id = node.deps[i];

          // Only wait if the dependency was ALSO a GPU operation
          // (CPU/GPU syncs are handled at the barrier level in smr.cc)
          if (dag.at(dep_id).target == ExecTarget::GPU) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, node_events[dep_id], 0));
          }
        }

        launch(node, stream, current_stream_idx);

        // FIXED: Increment by original_op_count for accurate metrics
        if (op_counter) {
          op_counter->fetch_add(node.original_op_count,
                                std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaEventRecord(node_events[op_id], stream));
        s_idx++;
      }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Sequential Execution Baseline
  void run_sequential(const std::unordered_map<uint64_t, DagNode>& dag,
                      std::atomic<int>* op_counter) {
    cudaStream_t seq_stream = streams_[0];

    // Because unordered_map loses generation order, we must manually
    // sort the operation IDs to ensure proper sequential state machine
    // execution.
    std::vector<uint64_t> sorted_keys;
    sorted_keys.reserve(dag.size());
    for (const auto& pair : dag) {
      sorted_keys.push_back(pair.first);
    }
    std::sort(sorted_keys.begin(), sorted_keys.end());

    for (uint64_t op_id : sorted_keys) {
      const DagNode& node = dag.at(op_id);

      // Skip CPU operations
      if (node.target != ExecTarget::GPU) continue;

      launch(node, seq_stream, 0);

      // Increment by original_op_count
      if (op_counter) {
        op_counter->fetch_add(node.original_op_count,
                              std::memory_order_relaxed);
      }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
  }

 private:
  void launch(const DagNode& node, cudaStream_t stream, int stream_idx) {
    float* d_out = device_mats_[node.operation.dest_mat_id_1.value()];

    if (node.has_fused_scalar) {
      launchFusedScalarMultiplyAndAdd(d_out, node.fused_alpha, node.fused_beta,
                                      rows_, cols_, stream);
    } else {
      switch (node.operation.type) {
        case OpType::SCALAR_ADD:
          launchAddScalar(d_out, (float)node.operation.scalar_param.value(),
                          rows_, cols_, stream);
          break;

        case OpType::SCALAR_SUB:
          launchSubtractScalar(d_out,
                               (float)node.operation.scalar_param.value(),
                               rows_, cols_, stream);
          break;

        case OpType::SCALAR_MULT:
          launchMultiplyScalar(d_out,
                               (float)node.operation.scalar_param.value(),
                               rows_, cols_, stream);
          break;

        case OpType::MAT_MULT: {
          float* d_mat_B = device_mats_[node.operation.dest_mat_id_2.value()];
          float* d_temp_result =
              stream_workspace_[stream_idx];  // use pre allocated buffer
          size_t size = rows_ * cols_ * sizeof(float);

          // SGEMM into temp buffer, then Async Copy back
          launchSgemm(d_out, d_mat_B, d_temp_result, rows_, cols_, cols_,
                      stream);
          CUDA_CHECK(cudaMemcpyAsync(d_out, d_temp_result, size,
                                     cudaMemcpyDeviceToDevice, stream));
          break;
        }

        case OpType::NEW_MAT_ADD: {
          float* d_param = d_op_params[node.operation.id];
          launchInPlaceMatrixAdd(d_out, d_param, rows_, cols_, stream);
          break;
        }

        case OpType::NEW_MAT_SUB: {
          float* d_param = d_op_params[node.operation.id];
          launchInPlaceMatrixSub(d_out, d_param, rows_, cols_, stream);
          break;
        }

        case OpType::NEW_MAT_MULT: {
          float* d_param = d_op_params[node.operation.id];
          float* d_temp_result = stream_workspace_[stream_idx];
          size_t size = rows_ * cols_ * sizeof(float);

          launchSgemm(d_out, d_param, d_temp_result, rows_, cols_, cols_,
                      stream);
          CUDA_CHECK(cudaMemcpyAsync(d_out, d_temp_result, size,
                                     cudaMemcpyDeviceToDevice, stream));
          break;
        }
        case OpType::ELEMAT_MULT: {
          float* d_param = d_op_params[node.operation.id];
          launchInPlaceElementwiseMatrixMult(d_out, d_param, rows_, cols_,
                                             stream);
          break;
        }

        default:
          break;
      }  // end switch
    }  // end if else
  }  // end launch

};  // end gpuexecutor class