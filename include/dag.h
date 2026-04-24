#pragma once

#include <cuda_runtime.h>

#include <fstream>
#include <map>
#include <set>
#include <vector>

#include "workload.h"
#include "kernels/common.cuh"

// enum class OpType : uint8_t {
//   SCALAR_ADD = 0,
//   SCALAR_SUB = 1,
//   SCALAR_MULT = 2,
//   MAT_ADD = 3,
//   MAT_SUB = 4,
//   MAT_MULT = 5,
//   NEW_MAT_ADD = 6,
//   NEW_MAT_SUB = 7,
//   NEW_MAT_MULT = 8
// };

struct DagNode {
  op operation;
  // std::vector<float> mat_data;
  float* d_mat_param = nullptr;
  std::set<uint64_t> deps;
  bool has_fused_scalar;
  int fused_scalar;
  uint64_t rows, cols;
};  // end dagnode struct

class DagGenerator {
 private:
  std::map<uint64_t, uint64_t> last_write;
  std::map<uint64_t, DagNode> dag;

  bool heavy_op(OpType type) {
    return type == OpType::MAT_MULT || type == OpType::MAT_ADD ||
        type == OpType::MAT_SUB || type == OpType::NEW_MAT_MULT ||
        type == OpType::NEW_MAT_ADD || type == OpType::NEW_MAT_SUB;
  }  // end heavy operation

 public:
  void build_dag(const std::vector<op>& log_slice) {
    for (auto& op : log_slice) {
      uint64_t targ_mat = op.dest_mat_id_1.value();

      if (last_write.count(targ_mat)) {
        uint64_t prev_op = last_write[targ_mat];
        DagNode& prev = dag[prev_op];

        // merge scalar ops
        if (op.type == prev.operation.type &&
            (op.type == OpType::SCALAR_ADD || op.type == OpType::SCALAR_SUB ||
             op.type == OpType::SCALAR_MULT) &&
            last_write.count(targ_mat)) {
          // [KAP325] I've assumed that when we do scalar subtrctions numbers
          // are passed in as positves
          if (prev.operation.type == OpType::SCALAR_ADD ||
              prev.operation.type == OpType::SCALAR_SUB) {
            prev.operation.scalar_param.value() += op.scalar_param.value();
          }  // end if

          if (prev.operation.type == OpType::SCALAR_MULT) {
            prev.operation.scalar_param.value() *= op.scalar_param.value();
          }  // end if

          last_write[op.id] = prev_op;
          continue;
        }  // end if

        // kernel fuxzion
        if ((op.type == OpType::SCALAR_ADD || op.type == OpType::SCALAR_MULT) &&
            heavy_op(prev.operation.type)) {
          uint64_t prev_op = last_write[targ_mat];
          DagNode& prev = dag[prev_op];

          int val = (op.type == OpType::SCALAR_SUB) ? -op.scalar_param.value()
                                                    : op.scalar_param.value();
          prev.fused_scalar += val;
          prev.has_fused_scalar = true;
          last_write[op.id] = prev_op;
          continue;
        }  // end if
      }  // end if

      DagNode node;
      node.operation = op;
      node.has_fused_scalar = false;
      node.fused_scalar = 0;

      if (last_write.count(op.dest_mat_id_1.value())) {
        node.deps.insert(last_write[op.dest_mat_id_1.value()]);
      }  // end if

      if (op.type == OpType::MAT_ADD || op.type == OpType::MAT_SUB ||
          op.type == OpType::MAT_MULT) {
        if (last_write.count(op.dest_mat_id_2.value())) {
          node.deps.insert(last_write[op.dest_mat_id_2.value()]);
        }  // end if
      }  // end if

      last_write[op.dest_mat_id_1.value()] = op.id;

      if (op.mat_param.has_value()) {
        // Process matrix parameter
        // TODO: needs to adjust to read from args instead of file
        const auto& mat = op.mat_param.value();
        
        node.rows = mat.num_rows;
        node.cols = mat.num_cols;
        
        size_t n_elements = node.rows * node.cols;
        size_t total_bytes = n_elements * sizeof(float);

        CUDA_CHECK(cudaMalloc(&node.d_mat_param, total_bytes));
        CUDA_CHECK(cudaMemcpy(node.d_mat_param, mat.data(), total_bytes, cudaMemcpyHostToDevice));
      }  // end if

      dag[op.id] = node;
    }  // end while
  }  // end bukld dag
  
  const std::map<uint64_t, DagNode>& get_dag() const {
    return dag;
  }  // end dag getter
};  // end class