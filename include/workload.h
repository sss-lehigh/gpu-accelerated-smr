#pragma once

#include <cstdint>
#include <optional>

#include "DenseMat.h"
#include "logging.h"

#define SCALAR_ADD_PERC 20
#define SCALAR_SUB_PERC 20
#define SCALAR_MULT_PERC 20
#define MAT_ADD_PERC 10
#define MAT_SUB_PERC 10
#define MAT_MULT_PERC 10

#define MATRIX_DIM 64
#define MAX_RANDOM_VALUE 1000

enum class OpType : uint8_t {
  SCALAR_ADD = 0,
  SCALAR_SUB = 1,
  SCALAR_MULT = 2,
  MAT_ADD = 3,
  MAT_SUB = 4,
  MAT_MULT = 5
};

struct op {
  uint64_t id;
  OpType type;
  std::optional<int> scalar_param;
  std::optional<DenseMat<int>> mat_param;
};

class WorkloadGenerator {
 private:
  std::vector<op> ops;

 public:
  WorkloadGenerator() = default;

  DenseMat<int> generateMatrix() {
    auto default_mat = DenseMat<int>(MATRIX_DIM, MATRIX_DIM);
    for (uint64_t i = 0; i < MATRIX_DIM; ++i) {
      for (uint64_t j = 0; j < MATRIX_DIM; ++j) {
        default_mat.set(i + 1, j + 1, rand() % MAX_RANDOM_VALUE);
      }
    }
    return default_mat;
  }

  std::vector<op> generate(uint64_t num_ops) {
    for (uint64_t i = 0; i < num_ops; ++i) {
      op new_op;
      new_op.id = i;
      int op_type_rand = rand() % 100;
      if (op_type_rand < SCALAR_ADD_PERC) {
        new_op.type = OpType::SCALAR_ADD;
        new_op.scalar_param = rand() % 100;
      } else if (op_type_rand < SCALAR_ADD_PERC + SCALAR_SUB_PERC) {
        new_op.type = OpType::SCALAR_SUB;
        new_op.scalar_param = rand() % 100;
      } else if (op_type_rand <
                 SCALAR_ADD_PERC + SCALAR_SUB_PERC + SCALAR_MULT_PERC) {
        new_op.type = OpType::SCALAR_MULT;
        new_op.scalar_param = rand() % 100;
      } else if (op_type_rand < SCALAR_ADD_PERC + SCALAR_SUB_PERC +
                                    SCALAR_MULT_PERC + MAT_ADD_PERC) {
        new_op.type = OpType::MAT_ADD;
        new_op.mat_param = generateMatrix();
      } else if (op_type_rand < SCALAR_ADD_PERC + SCALAR_SUB_PERC +
                                    SCALAR_MULT_PERC + MAT_ADD_PERC +
                                    MAT_SUB_PERC) {
        new_op.type = OpType::MAT_SUB;
        new_op.mat_param = generateMatrix();
      } else {
        new_op.type = OpType::MAT_MULT;
        new_op.mat_param = generateMatrix();
      }
      ops.push_back(new_op);
    }
    return ops;
  }

  void print(uint64_t start_idx, uint64_t end_idx) {
    LOGGING_ASSERT(start_idx < end_idx && end_idx <= ops.size(),
                   "Invalid index range for printing operations.");
    for (uint64_t i = start_idx; i < end_idx && i < ops.size(); ++i) {
      op& current_op = ops[i];
      LOGGING_INFO("Operation id {}:", current_op.id);
      switch (current_op.type) {
        case OpType::SCALAR_ADD: {
          LOGGING_INFO("Type: SCALAR_ADD, Scalar Param: {}",
                       current_op.scalar_param.value());
          break;
        }

        case OpType::SCALAR_SUB: {
          LOGGING_INFO("Type: SCALAR_SUB, Scalar Param: {}",
                       current_op.scalar_param.value());
          break;
        }

        case OpType::SCALAR_MULT: {
          LOGGING_INFO("Type: SCALAR_MULT, Scalar Param: {}",
                       current_op.scalar_param.value());
          break;
        }

        case OpType::MAT_ADD: {
          LOGGING_INFO("Type: MAT_ADD, Matrix Param: {}",
                       current_op.mat_param->ToString());
          break;
        }

        case OpType::MAT_SUB: {
          LOGGING_INFO("Type: MAT_SUB, Matrix Param: {}",
                       current_op.mat_param->ToString());
          break;
        }

        case OpType::MAT_MULT: {
          LOGGING_INFO("Type: MAT_MULT, Matrix Param: {}",
                       current_op.mat_param->ToString());
          break;
        }
      }
    }
  }
};