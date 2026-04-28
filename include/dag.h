#pragma once

#include <cstring>
#include <unordered_map>
#include <vector>

#include "workload.h"

// Defined for heterogeneous routing and benchmarking modes
enum class ExecTarget : uint8_t { CPU = 0, GPU = 1 };

enum class ExecMode : uint8_t {
  BASELINE_CPU = 0,
  BASELINE_GPU = 1,
  HYBRID = 2
};

struct DagNode {
  op operation;
  float* h_mat_param = nullptr;

  // Track the number of original log entries this node represents
  uint32_t original_op_count = 1;

  // Max 2 dependencies for binary math ops (replaces std::set)
  uint64_t deps[2];
  uint8_t dep_count = 0;

  // Forward edges and in_degree for fast O(V+E) level generation
  std::vector<uint64_t> children;
  int in_degree = 0;

  bool has_fused_scalar = false;
  float fused_alpha = 1.0f;
  float fused_beta = 0.0f;
  uint64_t rows = 0;
  uint64_t cols = 0;

  // Execution routing metadata
  int score = 0;
  ExecTarget target = ExecTarget::GPU;

  // Phase 2: Explicit DMA Pull Routing Flags
  bool pull_h2d_mat1 = false;
  bool pull_h2d_mat2 = false;
  bool pull_d2h_mat1 = false;
  bool pull_d2h_mat2 = false;

  void add_dep(uint64_t dep_id) {
    if (dep_count < 2) {
      for (int i = 0; i < dep_count; ++i) {
        if (deps[i] == dep_id) return;
      }
      deps[dep_count++] = dep_id;
      in_degree++;
    }
  }
};

class ExecutionGraph {
 private:
   int mat_dim_;
   ExecMode mode;
   std::unordered_map<uint64_t, uint64_t> last_write;
   std::unordered_map<uint64_t, DagNode> dag;

   // Tracks the physical location of matrices to penalize data ping-pong.
   // Do not clear this in reset()
   std::unordered_map<uint64_t, ExecTarget> mat_location;

   // Memory Arena for parameters
   std::vector<float> param_arena;
   size_t arena_offset = 0;

   bool heavy_op(OpType type)
   {
     return type == OpType::MAT_MULT || type == OpType::MAT_ADD ||
            type == OpType::MAT_SUB || type == OpType::NEW_MAT_MULT ||
            type == OpType::NEW_MAT_ADD || type == OpType::NEW_MAT_SUB ||
            type == OpType::ELEMAT_MULT;
   }

  void evaluate_and_route(DagNode& node) {
    // Hardware Performance Constants (Tune these to your specific cluster)
    constexpr float CPU_GFLOPS = 4000.0f;  // Estimated GFLOP/s for a single CPU core
    constexpr float GPU_GFLOPS = 10000.0f; // Estimated GFLOP/s for your GPU
    constexpr float GPU_LAUNCH_US = 5.0f;  // Kernel launch overhead in microseconds
    constexpr float PCIE_GBPS = 12.5f;     // PCIe Gen3/Gen4 bandwidth in GB/s

    // Calculate Theoretical FLOPs
    float total_flops = 0.0f;
    uint64_t num_elements = node.rows * node.cols;

    switch (node.operation.type) {
      case OpType::SCALAR_ADD:
      case OpType::SCALAR_SUB:
      case OpType::SCALAR_MULT:
      case OpType::MAT_ADD:
      case OpType::MAT_SUB:
      case OpType::NEW_MAT_ADD:
      case OpType::NEW_MAT_SUB:
      case OpType::ELEMAT_MULT:
        total_flops = static_cast<float>(num_elements);  // O(N) operations
        break;
      case OpType::MAT_MULT:
      case OpType::NEW_MAT_MULT:
        // SGEMM FLOPs: ~ 2 * M * N * K
        total_flops = 2.0f * static_cast<float>(node.rows) *
                      static_cast<float>(node.cols) *
                      static_cast<float>(node.cols);
        break;
    }

    // Base Execution Time in microseconds (us)
    float t_cpu = total_flops / (CPU_GFLOPS * 1000.0f);
    float t_gpu = GPU_LAUNCH_US + (total_flops / (GPU_GFLOPS * 1000.0f));

    // Data Locality / PCIe Penalties
    float bytes_to_transfer = static_cast<float>(num_elements * sizeof(float));
    float transfer_time_us = bytes_to_transfer / (PCIE_GBPS * 1000.0f);

    // Penalty A: Op brings a NEW matrix payload from Host (CPU) memory
    bool brings_new_host_mat = (node.operation.type == OpType::NEW_MAT_ADD ||
                                node.operation.type == OpType::NEW_MAT_SUB ||
                                node.operation.type == OpType::NEW_MAT_MULT);
    if (brings_new_host_mat) {
      t_gpu += transfer_time_us;  // GPU must fetch payload via PCIe
    }

    // Penalty B: Primary destination matrix is currently on the wrong device
    uint64_t dest_id_1 = node.operation.dest_mat_id_1.value();
    if (mat_location.count(dest_id_1)) {
      ExecTarget current_loc = mat_location[dest_id_1];
      if (current_loc == ExecTarget::CPU) {
        t_gpu += transfer_time_us;  // Costs a Host-to-Device copy
      } else if (current_loc == ExecTarget::GPU) {
        t_cpu += transfer_time_us;  // Costs a Device-to-Host copy
      }
    }

    // Penalty C: Second operand matrix is currently on the wrong device
    if (node.operation.type == OpType::MAT_ADD ||
        node.operation.type == OpType::MAT_SUB ||
        node.operation.type == OpType::MAT_MULT) {
      uint64_t dest_id_2 = node.operation.dest_mat_id_2.value();
      if (mat_location.count(dest_id_2)) {
        ExecTarget src_loc = mat_location[dest_id_2];
        if (src_loc == ExecTarget::CPU)
          t_gpu += transfer_time_us;
        else if (src_loc == ExecTarget::GPU)
          t_cpu += transfer_time_us;
      }
    }

    // Final Routing Decision
    node.target = (t_gpu < t_cpu) ? ExecTarget::GPU : ExecTarget::CPU;
    node.score = static_cast<int>(t_cpu - t_gpu);
  }

 public:
  ExecutionGraph(int mat_size, ExecMode m = ExecMode::HYBRID) : mat_dim_(mat_size), mode(m) {
    param_arena.resize(kNumProposals * mat_dim_ * mat_dim_);
  }

  void ingest_batch(const std::vector<const op*>& log_slice) {
    for (const op* op_ptr : log_slice) {
      op op = *op_ptr;
      if (op.type == OpType::SCALAR_SUB) {
        op.type = OpType::SCALAR_ADD;
        op.scalar_param.value() = -op.scalar_param.value();
      }

      uint64_t targ_mat = op.dest_mat_id_1.value();

      if (last_write.count(targ_mat)) {
        uint64_t prev_op = last_write[targ_mat];
        DagNode& prev = dag[prev_op];

        if (op.type == prev.operation.type &&
            (op.type == OpType::SCALAR_ADD || op.type == OpType::SCALAR_MULT)) {
          if (prev.operation.type == OpType::SCALAR_ADD) {
            prev.operation.scalar_param.value() += op.scalar_param.value();
          } else if (prev.operation.type == OpType::SCALAR_MULT) {
            prev.operation.scalar_param.value() *= op.scalar_param.value();
          }

          prev.original_op_count++;
          last_write[op.id] = prev_op;
          continue;
        }

        if ((op.type == OpType::SCALAR_ADD || op.type == OpType::SCALAR_MULT) &&
            heavy_op(prev.operation.type)) {
          if (op.type == OpType::SCALAR_ADD) {
            prev.fused_beta += op.scalar_param.value();
          } else if (op.type == OpType::SCALAR_MULT) {
            prev.fused_alpha *= op.scalar_param.value();
            prev.fused_beta *= op.scalar_param.value();
          }

          prev.has_fused_scalar = true;
          prev.original_op_count++;
          last_write[op.id] = prev_op;
          continue;
        }
      }

      DagNode node;
      node.operation = op;
      node.has_fused_scalar = false;

      // Build Dependencies AND Forward Edges
      if (last_write.count(op.dest_mat_id_1.value())) {
        uint64_t parent_id = last_write[op.dest_mat_id_1.value()];
        node.add_dep(parent_id);
        dag[parent_id].children.push_back(op.id);
      }

      if (op.type == OpType::MAT_ADD || op.type == OpType::MAT_SUB ||
          op.type == OpType::MAT_MULT) {
        if (last_write.count(op.dest_mat_id_2.value())) {
          uint64_t parent_id = last_write[op.dest_mat_id_2.value()];
          node.add_dep(parent_id);
          dag[parent_id].children.push_back(op.id);
        }
      }

      last_write[op.dest_mat_id_1.value()] = op.id;

      if (op.mat_param.has_value()) {
        const auto& mat = op.mat_param.value();
        node.rows = mat.num_rows;
        node.cols = mat.num_cols;

        size_t n_elements = node.rows * node.cols;
        node.h_mat_param = &param_arena[arena_offset];
        std::memcpy(node.h_mat_param, mat.data(), n_elements * sizeof(float));

        arena_offset += n_elements;
      } else {
        node.rows = mat_dim_;
        node.cols = mat_dim_;
      }

      // 1. Calculate score and assign target
      if (mode == ExecMode::BASELINE_CPU) {
        node.target = ExecTarget::CPU;
      } else if (mode == ExecMode::BASELINE_GPU) {
        node.target = ExecTarget::GPU;
      } else {
        evaluate_and_route(node);
      }

      // Resolve inter/intra-batch explicitly required DMA Pulls
      uint64_t mat_1 = op.dest_mat_id_1.value();
      if (mat_location.count(mat_1) && mat_location[mat_1] != node.target) {
        if (node.target == ExecTarget::GPU) node.pull_h2d_mat1 = true;
        if (node.target == ExecTarget::CPU) node.pull_d2h_mat1 = true;
      }
      mat_location[mat_1] = node.target; // Mat 1 is mutated, cache its new location

      if (op.type == OpType::MAT_ADD || op.type == OpType::MAT_SUB || op.type == OpType::MAT_MULT) {
        uint64_t mat_2 = op.dest_mat_id_2.value();
        if (mat_location.count(mat_2) && mat_location[mat_2] != node.target) {
          if (node.target == ExecTarget::GPU) node.pull_h2d_mat2 = true;
          if (node.target == ExecTarget::CPU) node.pull_d2h_mat2 = true;
          // Note: We do NOT update mat_location for Mat 2 because it's only read, 
          // meaning its latest mutated version still physically exists elsewhere.
        }
      }

      dag[op.id] = node;
    }
  }

  std::vector<std::vector<uint64_t>> generate_levels() {
    std::vector<std::vector<uint64_t>> lvls;
    std::vector<uint64_t> curr_lvl;

    for (auto& [id, node] : dag) {
      if (node.in_degree == 0) {
        curr_lvl.push_back(id);
      }
    }

    while (!curr_lvl.empty()) {
      lvls.push_back(curr_lvl);
      std::vector<uint64_t> next_level;

      for (uint64_t node_id : curr_lvl) {
        for (uint64_t child_id : dag[node_id].children) {
          dag[child_id].in_degree--;
          if (dag[child_id].in_degree == 0) {
            next_level.push_back(child_id);
          }
        }
      }
      curr_lvl = std::move(next_level);
    }
    return lvls;
  }

  const std::unordered_map<uint64_t, DagNode>& get_dag() const { return dag; }

  void reset() {
    ROMULUS_ASSERT(arena_offset <= param_arena.size(),
                   "Catch overflow for arena offset.");
    arena_offset = 0;
    dag.clear();
    last_write.clear();
  }
};