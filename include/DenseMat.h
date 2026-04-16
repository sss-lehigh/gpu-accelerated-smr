#pragma once

#include <cstdint>
#include <cstring>
#include <functional>

#include "../config.h"
#include "../include/panic.h"

/// A dense matrix, implemented as a 1D array
///
/// DenseMat is just a data type with some accessor methods.  It does not
/// implement any methods for doing linear algebra.  Indices are 1-based.
///
/// @tparam T The type of elements stored in the Matrix
template <typename T> class DenseMat {
public:
  static constexpr const char *name = "Dense Matrix";

  const uint64_t num_rows; // The number of rows in the Matrix (1-based)
  const uint64_t num_cols; // The number of columns in the Matrix (1-based)

protected:
  T *_vals; // A vector of vectors of values

public:
  /// Construct an empty DenseMat.  Every element will be initialized to `T()`.
  ///
  /// @param num_rows The number of rows.  It must be >0
  /// @param num_cols The number of columns.  It must be >0
  DenseMat(uint64_t num_rows, uint64_t num_cols)
      : num_rows(num_rows), num_cols(num_cols),
        _vals((T *)malloc(num_rows * num_cols * sizeof(T))) {
    if (BOUNDS_CHECK)
      if (num_rows < 1 || num_cols < 1)
        panic("Matrix dimensions must be >0");
    for (uint64_t i = 0; i < num_rows * num_cols; ++i)
      _vals[i] = T();
  }

  /// Copy-construct a DenseMat from another DenseMat
  ///
  /// @param other The DenseMat to copy
  DenseMat(const DenseMat &other)
      : num_rows(other.num_rows), num_cols(other.num_cols),
        _vals((T *)malloc(num_rows * num_cols * sizeof(T))) {
    memcpy(_vals, other._vals, num_rows * num_cols * sizeof(T));
  }

  /// Move-construct a DenseMat from another DenseMat
  ///
  /// @param other The DenseMat to move into this one
  DenseMat(DenseMat &&other) noexcept
      : num_rows(other.num_rows), num_cols(other.num_cols), _vals(other._vals) {
    // We moved everything over, so all that's left is to null other's array
    other._vals = nullptr;
  }

  // TODO: apply the rule of 0/3/5 to everything in the `mats` folder

  /// Copy assignment of one DenseMat into another is not valid, because
  /// `num_rows` and `num_cols` may not match
  ///
  /// @param other The DenseMat to copy-assign into this one
  DenseMat &operator=(const DenseMat &) = delete;

  /// Move-assign a DenseMat from another DenseMat is not valid, because
  /// `num_rows` and `num_cols` may not match.
  ///
  /// @param other The DenseMat to move-assign into this one
  DenseMat &operator=(DenseMat &&) = delete;

  /// Construct a square matrix
  ///
  /// @param dims The number of rows/columns.  It must be >0
  DenseMat(uint64_t dims) : DenseMat(dims, dims) {}

  /// Reclaim all memory associated with this DenseMat.
  ~DenseMat() {
    // NB: it could be null if we moved this DenseMat into another
    if (_vals != nullptr) {
      free(_vals);
      _vals = nullptr;
    }
  }

  /// Report the number of non-zeroes in the matrix
  uint64_t nonZeroes() {
    uint64_t count = 0;
    for (uint64_t i = 1; i <= num_rows; ++i)
      for (uint64_t j = 1; j <= num_cols; ++j)
        if (_vals[index(i, j)] != T())
          ++count;
    return count;
  }

  /// Run the provided function on each non-zero in the matrix
  ///
  /// NB: The main intent behind this function is to facilitate fast output
  ///
  /// @param op A lambda that takes row, column, value triples.
  void foreach_nonzero(std::function<void(uint64_t, uint64_t, const T &)> op) {
    for (uint64_t i = 1; i <= num_rows; ++i)
      for (uint64_t j = 1; j <= num_cols; ++j)
        if (_vals[index(i, j)] != T())
          op(i, j, _vals[index(i, j)]);
  }

  /// Return the value in the matrix at coordinate [row, col]
  ///
  /// @param row The 1-based row coordinate
  /// @param col The 1-based column coordinate
  T get(uint64_t row, uint64_t col) const {
    checkRowCol(row, col);
    return _vals[index(row, col)];
  }

  /// Set the value in the matrix at coordinate [row, col]
  ///
  /// @param row The 1-based row coordinate
  /// @param col The 1-based column coordinate
  /// @param val The value to put at [row, col]
  void set(uint64_t row, uint64_t col, T val) {
    checkRowCol(row, col);
    _vals[index(row, col)] = val;
  }

protected:
  /// Ensure the given row and column are within the dimensions of this Matrix
  ///
  /// @param row The 1-based row to check
  /// @param col The 1-based column to check
  void checkRowCol(uint64_t row, uint64_t col) const {
    if (BOUNDS_CHECK)
      if (row < 1 || col < 1 || row > num_rows || col > num_cols)
        panic("Coordinates out of range.");
  }

  /// Convert a 2-d matrix coordinate into an index into _vals
  ///
  /// @param i The row coordinate
  /// @param j The column coordinate
  /// @return  The index in _vals
  uint64_t index(uint64_t i, uint64_t j) const {
    return (i - 1) * num_cols + (j - 1);
  }
};
