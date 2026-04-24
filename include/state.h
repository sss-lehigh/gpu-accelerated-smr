#pragma once

#include <vector>
#include <random>
#include <type_traits>

#include "logging.h"
#include "DenseMat.h"

template <typename T>
class State {

  private:
    size_t num_matrix;
    std::vector<DenseMat<T>> matrices;

  public:
    State(int num = 5) : num_matrix{num} {
      matrices.reserve(num);
      for (size_t i = 0; i < num; ++i) {
        // emplace_back passes these arguments directly to DenseMat(uint64_t, uint64_t)
        matrices.emplace_back(ROWS, COLS); 
      }
    }
    State(std::vector<DenseMat<T>> matrices) : matrices(matrices), num_matrix(matrices.size()) {}

    DenseMat<T> getMatrix(size_t index) const {
      if (index >= matrices.size()) {
        LOGGING_FATAL("Matrix index out of range.");
      }
      return matrices[index];
    }


    void populate_random_state_matrix(T min_val = static_cast<T>(0), T max_val = static_cast<T>(1)) {
        // Initialize the standard Mersenne Twister PRNG
        std::random_device rd;
        std::mt19937 gen(rd());

        // C++17 constexpr if ensures the compiler only instantiates the valid distribution for T
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dist(min_val, max_val);
            fill_matrices(dist, gen);
        } else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dist(min_val, max_val);
            fill_matrices(dist, gen);
        } else {
            // Fallback or static assertion if T is a complex number or unsupported type
            static_assert(std::is_arithmetic_v<T>, "Type T must be an arithmetic type to generate random numbers natively.");
        }
    }

private:
  // Helper template to avoid duplicating the nested loop logic for different distribution types
  template <typename DistType, typename GeneratorType>
  void fill_matrices(DistType& dist, GeneratorType& gen) {
      for (auto& matrix : matrices) {
          // Assuming DenseMat has rows(), cols(), and operator()(r, c)
          for (size_t r = 1; r <= matrix.num_rows; ++r) {
              for (size_t c = 1; c <= matrix.num_cols; ++c) {
                  matrix.set(r, c, dist(gen));
              }
          }
      }
  }

};