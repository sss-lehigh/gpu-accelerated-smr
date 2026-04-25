#pragma once

#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <cstring>

#include "workload.h"

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
  float* h_mat_param = nullptr;
  std::set<uint64_t> deps;
  bool has_fused_scalar = false;
  float fused_alpha = 1.0f; 
  float fused_beta = 0.0f;
  uint64_t rows = 0;
  uint64_t cols = 0;
};

class DagGenerator {
 private:
  std::map<uint64_t, uint64_t> last_write;
  std::map<uint64_t, DagNode> dag;

  bool heavy_op(OpType type) {
    return type == OpType::MAT_MULT || type == OpType::MAT_ADD ||
           type == OpType::MAT_SUB || type == OpType::NEW_MAT_MULT ||
           type == OpType::NEW_MAT_ADD || type == OpType::NEW_MAT_SUB ||
           type == OpType::ELEMAT_MULT;
  }  // end heavy operation

 public: 
  // RESTORED: Destructor to prevent memory leaks
  ~DagGenerator() {
    for (auto& pair : dag) {
      if (pair.second.h_mat_param != nullptr) {
        delete[] pair.second.h_mat_param;
        pair.second.h_mat_param = nullptr;
      }
    }
  }

  void build_dag(const std::vector<op>& log_slice) {
    // FIXED: Use 'auto op' to make a mutable copy, not a const reference
    for (auto op : log_slice) {
      
      // RESTORED: Normalize Subtractions to Additions
      if (op.type == OpType::SCALAR_SUB) {
        op.type = OpType::SCALAR_ADD;
        op.scalar_param.value() = -op.scalar_param.value();
      }

      uint64_t targ_mat = op.dest_mat_id_1.value();

      // RESTORED: The scope block that fetches the previous operation
      if (last_write.count(targ_mat)) {
        uint64_t prev_op = last_write[targ_mat];
        DagNode& prev = dag[prev_op];

        // merge scalar ops
        if (op.type == prev.operation.type &&
            (op.type == OpType::SCALAR_ADD || op.type == OpType::SCALAR_MULT)) {

          if (prev.operation.type == OpType::SCALAR_ADD) {
            prev.operation.scalar_param.value() += op.scalar_param.value();
          } else if (prev.operation.type == OpType::SCALAR_MULT) {
            prev.operation.scalar_param.value() *= op.scalar_param.value();
          }

          last_write[op.id] = prev_op;
          continue;
        }  // end if

        // kernel fusion
        if ((op.type == OpType::SCALAR_ADD || op.type == OpType::SCALAR_MULT) &&
            heavy_op(prev.operation.type)) {

          if (op.type == OpType::SCALAR_ADD) {
              prev.fused_beta += op.scalar_param.value();
          } else if (op.type == OpType::SCALAR_MULT) {
              prev.fused_alpha *= op.scalar_param.value();
              prev.fused_beta *= op.scalar_param.value(); 
          }

          prev.has_fused_scalar = true;
          last_write[op.id] = prev_op;
          continue;
        }  // end if
      }  // end if

      DagNode node;
      node.operation = op;
      node.has_fused_scalar = false;

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
        const auto& mat = op.mat_param.value();
        
        node.rows = mat.num_rows;
        node.cols = mat.num_cols;
        
        size_t n_elements = node.rows * node.cols;
        size_t total_bytes = n_elements * sizeof(float);

        node.h_mat_param = new float[n_elements];
        std::memcpy(node.h_mat_param, mat.data(), total_bytes);
      }

      dag[op.id] = node;
    }  // end for
  }  // end build_dag
  
  const std::map<uint64_t, DagNode>& get_dag() const {
    return dag;
  }
};  // end class