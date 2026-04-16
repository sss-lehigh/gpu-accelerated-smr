#pragma once

#include <vector>

#include "logging.h"
#include "DenseMat.h"

template <typename T>
class State {

  private:
    std::vector<DenseMat<T>> matrices;
  
  public:
    State() = default;
    State(std::vector<DenseMat<T>> matrices) : matrices(matrices) {}

    DenseMat<T> getMatrix(int index) {
      if (index < 0 || index >= matrices.size()) {
        LOGGING_FATAL("Matrix index out of range.");
      }
      return matrices[index];
    }

};