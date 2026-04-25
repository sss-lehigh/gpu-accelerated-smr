#pragma once

#include <cstdint>
#include <fstream>
#include <optional>

#include "DenseMat.h"
#include "romulus/logging.h"

#define SCALAR_ADD_PERC 20
#define SCALAR_SUB_PERC 20
#define SCALAR_MULT_PERC 10
#define MAT_ADD_PERC 10
#define MAT_SUB_PERC 10
#define MAT_MULT_PERC 5
#define NEW_MAT_ADD_PERC 5
#define NEW_MAT_SUB_PERC 5
#define NEW_MAT_MULT_PERC 5
#define ELE_MAT_MULT 10

#define MATRIX_DIM 2
#define MAX_RANDOM_VALUE 1000

enum class OpType : uint8_t {
  SCALAR_ADD = 0,
  SCALAR_SUB = 1,
  SCALAR_MULT = 2,
  MAT_ADD = 3,
  MAT_SUB = 4,
  MAT_MULT = 5,
  NEW_MAT_ADD = 6,
  NEW_MAT_SUB = 7,
  NEW_MAT_MULT = 8, 
  ELEMAT_MULT = 9
};

struct op {
  uint64_t id;
  OpType type;
  std::optional<uint64_t> dest_mat_id_1;
  std::optional<uint64_t> dest_mat_id_2;
  // Will only be populated if we have a "new" prefix, indicating that this
  // will have additional matrix as the parameter instead of index
  std::optional<float> scalar_param;
  std::optional<DenseMat<float>> mat_param;
};

struct SerializedOp {
  uint64_t id;
  OpType type;
  uint64_t dest_mat_id_1;
  uint64_t dest_mat_id_2;
  float scalar_param;
  bool has_mat_param;
  // uint64_t rows;
  // uint64_t cols;
};  // end serialized op struc t

class WorkloadGenerator {
 private:
  std::vector<op> ops;

 public:
  WorkloadGenerator() = default;

  DenseMat<float> generateMatrix() {
    auto default_mat = DenseMat<float>(MATRIX_DIM, MATRIX_DIM);
    for (uint64_t i = 0; i < MATRIX_DIM; ++i) {
      for (uint64_t j = 0; j < MATRIX_DIM; ++j) {
        default_mat.set(i + 1, j + 1, static_cast<float>(rand() % MAX_RANDOM_VALUE));
      }
    }
    return default_mat;
  }

  DenseMat<float> generateMatrix(uint64_t dim) {
    auto default_mat = DenseMat<float>(dim, dim);
    for (uint64_t i = 0; i < dim; ++i) {
      for (uint64_t j = 0; j < dim; ++j) {
        default_mat.set(i + 1, j + 1, static_cast<float>(rand() % MAX_RANDOM_VALUE));
      }
    }
    return default_mat;
  }

  std::vector<op> generate(uint64_t num_ops, uint64_t num_mats = 1) {
    for (uint64_t i = 0; i < num_ops; ++i) {
      op new_op;
      new_op.id = i;
      int op_type_rand = rand() % 100;
      if (op_type_rand < SCALAR_ADD_PERC) {
        new_op.type = OpType::SCALAR_ADD;
        new_op.dest_mat_id_1 = rand() % num_mats;
        new_op.scalar_param = static_cast<float>(rand() % MAX_RANDOM_VALUE);
        // rest of parameters are ignored...
      } else if (op_type_rand < SCALAR_ADD_PERC + SCALAR_SUB_PERC) {
        new_op.type = OpType::SCALAR_SUB;
        new_op.dest_mat_id_1 = rand() % num_mats;
        new_op.scalar_param = static_cast<float>(rand() % MAX_RANDOM_VALUE);
        // rest of parameters are ignored...
      } else if (op_type_rand <
                 SCALAR_ADD_PERC + SCALAR_SUB_PERC + SCALAR_MULT_PERC) {
        new_op.type = OpType::SCALAR_MULT;
        new_op.dest_mat_id_1 = rand() % num_mats;
        new_op.scalar_param = static_cast<float>(rand() % MAX_RANDOM_VALUE);
        // rest of parameters are ignored...
      } else if (op_type_rand < SCALAR_ADD_PERC + SCALAR_SUB_PERC +
                                    SCALAR_MULT_PERC + MAT_ADD_PERC) {
        new_op.type = OpType::MAT_ADD;
        new_op.dest_mat_id_1 = rand() % num_mats;
        new_op.dest_mat_id_2 = rand() % num_mats;
        // rest of parameters are ignored...
      } else if (op_type_rand < SCALAR_ADD_PERC + SCALAR_SUB_PERC +
                                    SCALAR_MULT_PERC + MAT_ADD_PERC +
                                    MAT_SUB_PERC) {
        new_op.type = OpType::MAT_SUB;
        new_op.dest_mat_id_1 = rand() % num_mats;
        new_op.dest_mat_id_2 = rand() % num_mats;
        // rest of parameters are ignored...
      } else if (op_type_rand < SCALAR_ADD_PERC + SCALAR_SUB_PERC +
                                    SCALAR_MULT_PERC + MAT_ADD_PERC +
                                    MAT_SUB_PERC + MAT_MULT_PERC) {
        new_op.type = OpType::MAT_MULT;
        new_op.dest_mat_id_1 = rand() % num_mats;
        new_op.dest_mat_id_2 = rand() % num_mats;
        // rest of parameters are ignored...
      } else if (op_type_rand < SCALAR_ADD_PERC + SCALAR_SUB_PERC +
                                    SCALAR_MULT_PERC + MAT_ADD_PERC +
                                    MAT_SUB_PERC + MAT_MULT_PERC +
                                    NEW_MAT_ADD_PERC) {
        new_op.type = OpType::NEW_MAT_ADD;
        new_op.dest_mat_id_1 = rand() % num_mats;
        new_op.mat_param = generateMatrix();
        // rest of parameters are ignored...
      } else if (op_type_rand < SCALAR_ADD_PERC + SCALAR_SUB_PERC +
                                    SCALAR_MULT_PERC + MAT_ADD_PERC +
                                    MAT_SUB_PERC + MAT_MULT_PERC +
                                    NEW_MAT_ADD_PERC + NEW_MAT_SUB_PERC) {
        new_op.type = OpType::NEW_MAT_SUB;
        new_op.dest_mat_id_1 = rand() % num_mats;
        new_op.mat_param = generateMatrix();
        // rest of parameters are ignored...
      } else if (op_type_rand < SCALAR_ADD_PERC + SCALAR_SUB_PERC +
                                    SCALAR_MULT_PERC + MAT_ADD_PERC +
                                    MAT_SUB_PERC + MAT_MULT_PERC +
                                    NEW_MAT_ADD_PERC + NEW_MAT_SUB_PERC + NEW_MAT_MULT_PERC) {
        new_op.type = OpType::NEW_MAT_MULT;
        new_op.dest_mat_id_1 = rand() % num_mats;
        new_op.mat_param = generateMatrix();
        // rest of parameters are ignored...
      } else {
        new_op.type = OpType::ELEMAT_MULT;
        new_op.dest_mat_id_1 = rand() % num_mats;
        new_op.mat_param = generateMatrix();
        // rest of parameters are ignored...
      }
      ops.push_back(new_op);
    }
    return ops;
  }

  void write_log(const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);

    for (const auto& op : ops) {
      SerializedOp sop;
      sop.id = op.id;
      sop.type = op.type;
      sop.dest_mat_id_1 = op.dest_mat_id_1.value_or(-1);
      sop.dest_mat_id_2 = op.dest_mat_id_2.value_or(-1);
      sop.scalar_param = op.scalar_param.value_or(0);
      sop.has_mat_param = op.mat_param.has_value();

      ofs.write(reinterpret_cast<char*>(&sop), sizeof(SerializedOp));

      if (sop.has_mat_param) {
        auto& mat = op.mat_param.value();
        uint64_t rows = mat.num_rows;
        uint64_t cols = mat.num_cols;

        ofs.write(reinterpret_cast<const char*>(&rows), sizeof(uint64_t));
        ofs.write(reinterpret_cast<const char*>(&cols), sizeof(uint64_t));

        size_t size = rows * cols * sizeof(float);

        ofs.write(reinterpret_cast<const char*>(mat.data()), size);
      }  // end if
    }  // end for

    ofs.close();
    ROMULUS_INFO("Dummy log generated at: {}", path);
  }  // end write log fcn

  void print(uint64_t start_idx, uint64_t end_idx) {
    ROMULUS_ASSERT(start_idx < end_idx && end_idx <= ops.size(),
                   "Invalid index range for printing operations.");
    ROMULUS_INFO("########### Workload ###########");
    for (uint64_t i = start_idx; i < end_idx && i < ops.size(); ++i) {
      op& current_op = ops[i];
      ROMULUS_INFO("Operation id {}:", current_op.id);
      switch (current_op.type) {
        case OpType::SCALAR_ADD: {
          ROMULUS_INFO("Type: SCALAR_ADD, Scalar Param: {}, State Mat ID: {}",
                       current_op.scalar_param.value(),
                       current_op.dest_mat_id_1.value());
          break;
        }

        case OpType::SCALAR_SUB: {
          ROMULUS_INFO("Type: SCALAR_SUB, Scalar Param: {}, State Mat ID: {}",
                       current_op.scalar_param.value(),
                       current_op.dest_mat_id_1.value());
          break;
        }

        case OpType::SCALAR_MULT: {
          ROMULUS_INFO("Type: SCALAR_MULT, Scalar Param: {}, State Mat ID: {}",
                       current_op.scalar_param.value(),
                       current_op.dest_mat_id_1.value());
          break;
        }

        case OpType::MAT_ADD: {
          ROMULUS_INFO("Type: MAT_ADD, M1_ID: {}, M2_ID: {}",
                       current_op.dest_mat_id_1.value(),
                       current_op.dest_mat_id_2.value());
          break;
        }

        case OpType::MAT_SUB: {
          ROMULUS_INFO("Type: MAT_SUB, M1_ID: {}, M2_ID: {}",
                       current_op.dest_mat_id_1.value(),
                       current_op.dest_mat_id_2.value());
          break;
        }

        case OpType::MAT_MULT: {
          ROMULUS_INFO("Type: MAT_MULT, M1_ID: {}, M2_ID: {}",
                       current_op.dest_mat_id_1.value(),
                       current_op.dest_mat_id_2.value());
          break;
        }

        case OpType::NEW_MAT_ADD: {
          ROMULUS_INFO("Type: NEW_MAT_ADD, Matrix ID: {}, Incoming Matrix: {}",
                       current_op.dest_mat_id_1.value(),
                       current_op.mat_param->ToString());
          break;
        }

        case OpType::NEW_MAT_SUB: {
          ROMULUS_INFO("Type: NEW_MAT_SUB, Matrix ID: {}, Incoming Matrix: {}",
                       current_op.dest_mat_id_1.value(),
                       current_op.mat_param->ToString());
          break;
        }

        case OpType::NEW_MAT_MULT: {
          ROMULUS_INFO("Type: NEW_MAT_MULT, Matrix ID: {}, Incoming Matrix: {}",
                       current_op.dest_mat_id_1.value(),
                       current_op.mat_param->ToString());
          break;
        }

        case OpType::ELEMAT_MULT: {
          ROMULUS_INFO("Type: ELEMAT_MULT, Matrix ID: {}, Incoming Matrix: {}",
                       current_op.dest_mat_id_1.value(),
                       current_op.mat_param->ToString());
          break;
        }
      }
      ROMULUS_INFO("-----------------------------------");
    }
  }
};