#pragma once

#include <cstring>

#include "../config.h"
#include "../include/panic.h"

/// A dense vector, implemented as a 1d array
///
/// DenseVec is just a data type with some accessor methods.  it does not
/// implement any methods for doing linear algebra.
///
/// @tparam T The type of elements stored in the vector
template <typename T> class DenseVec {
public:
  static constexpr const char *name = "Dense Vector";

  const uint64_t size; // The number of elements in the vector (1-based)

protected:
  T *_vals; // A vector of values

public:
  /// Construct an empty DenseVec.  Every element will be initialized to `T()`.
  ///
  /// @param size The size of the vector.  It must be >0
  DenseVec(uint64_t size) : size(size), _vals((T *)malloc(size * sizeof(T))) {
    if (BOUNDS_CHECK)
      if (size < 1)
        panic("Vector size must be >0");
    for (uint64_t i = 0; i < size; ++i)
      _vals[i] = T();
  }

  /// Copy-construct a DenseVec from another DenseVec
  ///
  /// @param other The DenseVec to copy
  DenseVec(const DenseVec &other)
      : size(other.size), _vals((T *)malloc(size * sizeof(T))) {
    memcpy(_vals, other._vals, size * sizeof(T));
  }

  /// Move-construct a DenseVec from another DenseVec
  ///
  /// @param other The DenseVec to move into this one
  DenseVec(DenseVec &&other) noexcept : size(other.size), _vals(other._vals) {
    // We moved everything over, so all that's left is to null other's array
    other._vals = nullptr;
  }

  /// Copy assignment of one DenseVec into another is not valid, because
  /// `size` may not match
  ///
  /// @param other The DenseVec to copy-assign into this one
  DenseVec &operator=(const DenseVec &) = delete;

  /// Move-assign a DenseVec from another DenseVec is not valid, because
  /// `num_rows` and `num_cols` may not match.
  ///
  /// @param other The DenseVec to move-assign into this one
  DenseVec &operator=(DenseVec &&) = delete;

  /// Reclaim all memory associated with this DenseVec.
  ~DenseVec() {
    // NB: it could be null if we moved this DenseVec into another
    if (_vals != nullptr) {
      free(_vals);
      _vals = nullptr;
    }
  }

  /// Report the number of non-zeroes in the vector
  uint64_t nonZeroes() const {
    uint64_t count = 0;
    for (uint64_t i = 0; i < size; ++i)
      if (_vals[i] != T())
        ++count;
    return count;
  }

  /// Clear the vector
  void clear() { memset(_vals, 0, size * sizeof(T)); }

  /// Return the value in the vector at index `idx`
  ///
  /// @param idx The 1-based index
  T get(uint64_t idx) const {
    checkIdx(idx);
    return _vals[idx - 1];
  }

  /// Set the value in the vector at index `idx`
  ///
  /// @param idx The 1-based index
  /// @param val The value to put at [row, col]
  void set(uint64_t idx, T val) {
    checkIdx(idx);
    _vals[idx - 1] = val;
  }

protected:
  /// Ensure the given index is within the size of this Vector
  ///
  /// @param idx The 1-based index to check
  void checkIdx(uint64_t idx) const {
    if (BOUNDS_CHECK)
      if (idx < 1 || idx > size)
        panic("Index out of range.");
  }

public: // Friend methods for doing linear algebra
  template <typename X>
  friend bool operator==(const DenseVec<X> &, const DenseVec<X> &);
};
