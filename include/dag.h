#pragma once

#include <cstring>
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
  std::map<uint64_t, uint64_t> last_write_;
  std::map<uint64_t, DagNode> dag_;
  std::vector<int> op_scores_;

  bool heavy_op(OpType type) {
    return type == OpType::MAT_MULT || type == OpType::MAT_ADD ||
           type == OpType::MAT_SUB || type == OpType::NEW_MAT_MULT ||
           type == OpType::NEW_MAT_ADD || type == OpType::NEW_MAT_SUB ||
           type == OpType::ELEMAT_MULT;
  }  // end heavy operation


  // NOTE: I used AI to generated the following score function: 
  // TODO: We will want to adjust this function experimentally...
  // 
  // 
  // Scores of (0,50] will be send to cpu, while scores of (50,100] will be sent
  // to gpu Function to generate a score from 0-100 based on the parameters of
  // the given
  // operation.
  //
  // Factors considered:
  //   - Operation type (arithmetic intensity differs: SGEMM ≫ elementwise ≫
  //   scalar)
  //   - Matrix size (PCIe + launch overhead amortizes only above a threshold)
  //   - Whether the op carries a "new" host-resident matrix (NEW_MAT_*), which
  //     forces a host-to-device transfer that hurts GPU economics for small ops
  //
  // Scoring philosophy: each op has a per-element GPU benefit (arithmetic
  // intensity proxy) and a fixed GPU cost (launch overhead, plus optional H2D
  // transfer). The score reflects whether benefit × work_size beats cost.
  //
  // (0, 50]  -> CPU
  // (50, 100] -> GPU
  int get_op_score(const op& operation) {
    // Per-element GPU "benefit" by op type. Roughly proportional to the FLOPs
    // and memory parallelism the GPU can extract per element.
    //   - Scalar ops: 1 FLOP per element, memory-bound -> low benefit
    //   - Elementwise matrix ops: 1 FLOP per element, two reads -> low-medium
    //   - SGEMM: O(N) FLOPs per element, cache-blocked -> very high benefit
    float per_element_benefit = 0.0f;
    switch (operation.type) {
      case OpType::SCALAR_ADD:
      case OpType::SCALAR_SUB:
      case OpType::SCALAR_MULT:
        per_element_benefit = 1.0f;  // streaming, GPU barely helps until huge
        break;
      case OpType::MAT_ADD:
      case OpType::MAT_SUB:
      case OpType::NEW_MAT_ADD:
      case OpType::NEW_MAT_SUB:
      case OpType::ELEMAT_MULT:
        per_element_benefit = 2.0f;  // elementwise, two operand reads
        break;
      case OpType::MAT_MULT:
      case OpType::NEW_MAT_MULT:
        per_element_benefit = 16.0f;  // SGEMM is the GPU's natural strength
        break;
    }

    // Problem size proxy. We don't have the matrix dimensions on the op
    // directly, but a NEW_MAT_* op carries the operand and we can read it; for
    // dest-only ops we fall back to the global matrix size constants.
    uint64_t num_elements;
    if (operation.mat_param.has_value()) {
      const auto& m = operation.mat_param.value();
      num_elements =
          static_cast<uint64_t>(m.rows()) * static_cast<uint64_t>(m.cols());
    } else {
      num_elements = static_cast<uint64_t>(ROWS) * static_cast<uint64_t>(COLS);
    }

    // Total GPU benefit (arbitrary units; the threshold below absorbs the
    // scale).
    float benefit = per_element_benefit * static_cast<float>(num_elements);

    // Fixed GPU cost: kernel launch (~5-10µs on V100, modeled in arbitrary
    // units) plus an H2D transfer penalty if the op carries a host-resident
    // matrix.
    constexpr float LAUNCH_COST = 50000.0f;    // baseline overhead per kernel
    constexpr float H2D_COST_PER_ELEM = 0.5f;  // PCIe transfer cost per element

    float cost = LAUNCH_COST;
    bool needs_h2d = (operation.type == OpType::NEW_MAT_ADD ||
                      operation.type == OpType::NEW_MAT_SUB ||
                      operation.type == OpType::NEW_MAT_MULT);
    if (needs_h2d) {
      cost += H2D_COST_PER_ELEM * static_cast<float>(num_elements);
    }

    // Map benefit/cost ratio to score in [0, 100], centered at 50 (the
    // CPU/GPU threshold). A ratio of 1.0 means break-even -> score = 50.
    // We use a logistic-style squash so unbounded ratios stay in range.
    float ratio = benefit / cost;
    float score = 100.0f * (ratio / (ratio + 1.0f));

    // Clamp defensively (the formula is already bounded but rounding could
    // nudge).
    if (score < 0.0f) score = 0.0f;
    if (score > 100.0f) score = 100.0f;

    return static_cast<int>(score);
  }

 public:
  explicit DagGenerator(const std::vector<op>& log_slice) : op_scores_(log_slice.size()) {
    // FIXED: Use 'auto op' to make a mutable copy, not a const reference
    for (auto op : log_slice) {
      // RESTORED: Normalize Subtractions to Additions
      if (op.type == OpType::SCALAR_SUB) {
        op.type = OpType::SCALAR_ADD;
        op.scalar_param.value() = -op.scalar_param.value();
      }

      uint64_t targ_mat = op.dest_mat_id_1.value();

      // RESTORED: The scope block that fetches the previous operation
      if (last_write_.count(targ_mat)) {
        uint64_t prev_op = last_write_[targ_mat];
        DagNode& prev = dag_[prev_op];

        // merge scalar ops
        if (op.type == prev.operation.type &&
            (op.type == OpType::SCALAR_ADD || op.type == OpType::SCALAR_MULT)) {
          if (prev.operation.type == OpType::SCALAR_ADD) {
            prev.operation.scalar_param.value() += op.scalar_param.value();
          } else if (prev.operation.type == OpType::SCALAR_MULT) {
            prev.operation.scalar_param.value() *= op.scalar_param.value();
          }

          last_write_[op.id] = prev_op;
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
          last_write_[op.id] = prev_op;
          continue;
        }  // end if
      }  // end if

      DagNode node;
      node.operation = op;
      node.has_fused_scalar = false;

      if (last_write_.count(op.dest_mat_id_1.value())) {
        node.deps.insert(last_write_[op.dest_mat_id_1.value()]);
      }  // end if

      if (op.type == OpType::MAT_ADD || op.type == OpType::MAT_SUB ||
          op.type == OpType::MAT_MULT) {
        if (last_write_.count(op.dest_mat_id_2.value())) {
          node.deps.insert(last_write_[op.dest_mat_id_2.value()]);
        }  // end if
      }  // end if

      last_write_[op.dest_mat_id_1.value()] = op.id;

      if (op.mat_param.has_value()) {
        const auto& mat = op.mat_param.value();

        node.rows = mat.num_rows;
        node.cols = mat.num_cols;

        size_t n_elements = node.rows * node.cols;
        size_t total_bytes = n_elements * sizeof(float);

        node.h_mat_param = new float[n_elements];
        std::memcpy(node.h_mat_param, mat.data(), total_bytes);
      }

      dag_[op.id] = node;
      op_scores_[op.id] = get_op_score(op);
    }  // end for
  }
  // RESTORED: Destructor to prevent memory leaks
  ~DagGenerator() {
    for (auto& pair : dag_) {
      if (pair.second.h_mat_param != nullptr) {
        delete[] pair.second.h_mat_param;
        pair.second.h_mat_param = nullptr;
      }
    }
  }

  const std::map<uint64_t, DagNode>& get_dag() const { return dag_; }

  // reset the dag
  void reset() {
    // Free all dynamically allocated host memory for the current batch
    for (auto& pair : dag_) {
      if (pair.second.h_mat_param != nullptr) {
        delete[] pair.second.h_mat_param;
        pair.second.h_mat_param = nullptr;
      }
    }

    // Clear the dependency tracker and the DAG container
    dag_.clear();
    last_write_.clear();
  }
};  // end class