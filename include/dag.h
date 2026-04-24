#pragma once

#include <cuda_runtime.h>

#include <fstream>
#include <map>
#include <set>
#include <vector>

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
  SerializedOp op;
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
      uint64_t targ_mat = op.dest_mat_id_1;

      if (last_write.count(targ_mat)) {
        uint64_t prev_op = last_write[targ_mat];
        DagNode& prev = dag[prev_op];

        // merge scalar ops
        if (op.type == prev.op.type &&
            (op.type == OpType::SCALAR_ADD || op.type == OpType::SCALAR_SUB ||
             op.type == OpType::SCALAR_MULT) &&
            last_write.count(targ_mat)) {
          // [KAP325] I've assumed that when we do scalar subtrctions numbers
          // are passed in as positves
          if (prev.op.type == OpType::SCALAR_ADD ||
              prev.op.type == OpType::SCALAR_SUB) {
            prev.op.scalar_param += op.scalar_param;
          }  // end if

          if (prev.op.type == OpType::SCALAR_MULT) {
            prev.op.scalar_param *= op.scalar_param;
          }  // end if

          last_write[op.id] = prev_op;
          continue;
        }  // end if

        // kernel fuxzion
        if ((op.type == OpType::SCALAR_ADD || op.type == OpType::SCALAR_MULT) &&
            heavy_op(prev.op.type)) {
          uint64_t prev_op = last_write[targ_mat];
          DagNode& prev = dag[prev_op];

          int val = (op.type == OpType::SCALAR_SUB) ? -op.scalar_param
                                                    : op.scalar_param;
          prev.fused_scalar += val;
          prev.has_fused_scalar = true;
          last_write[op.id] = prev_op;
          continue;
        }  // end if
      }  // end if

      DagNode node;
      node.op = op;
      node.has_fused_scalar = false;
      node.fused_scalar = 0;

      if (last_write.count(op.dest_mat_id_1)) {
        node.deps.insert(last_write[op.dest_mat_id_1]);
      }  // end if

      if (op.type == OpType::MAT_ADD || op.type == OpType::MAT_SUB ||
          op.type == OpType::MAT_MULT) {
        if (last_write.count(op.dest_mat_id_2)) {
          node.deps.insert(last_write[op.dest_mat_id_2]);
        }  // end if
      }  // end if

      last_write[op.dest_mat_id_1] = op.id;

      if (op.has_mat_param) {
        uint64_t r, c;
        log.read(reinterpret_cast<char*>(&r), sizeof(uint64_t));
        log.read(reinterpret_cast<char*>(&c), sizeof(uint64_t));

        node.rows = r;
        node.cols = c;

        std::vector<float> float_data(r * c);
        log.read(reinterpret_cast<char*>(float_data.data()),
                 r * c * sizeof(float));

        cudaMalloc(&node.d_mat_param, r * c * sizeof(float));
        cudaMemcpy(node.d_mat_param, float_data.data(), r * c * sizeof(float),
                   cudaMemcpyHostToDevice);

        // size_t n_rows = r*c;
        // size_t total_bytes = n_rows*sizeof(float);

        // node.mat_data.resize(r * c * sizeof(float));
        // log.read(reinterpret_cast<char*>(node.mat_data.data()),
        // node.mat_data.size());
      }  // end iof

      dag[op.id] = node;
    }  // end while
  }  // end bukld dag

  const std::map<uint64_t, DagNode>& get_dag() const {
    return dag;
  }  // end dag getter
};  // end class