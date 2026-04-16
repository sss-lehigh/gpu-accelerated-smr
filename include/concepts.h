#pragma once

#include <concepts>
#include <cstdint>

/// The MATRIX concept: a minimal description of what we expect from anything
/// that purports to be a matrix
template <template <typename> typename M, typename T>
concept MATRIX = requires(M<T> m, T v, uint64_t r, uint64_t c) {
  { m.name };                         // A static field named `name`
  { m.num_rows };                     // A field named `num_rows`
  { m.num_cols };                     // A field named `num_cols`
  { m.get(r, c) } -> std::same_as<T>; // Method to get values by row/column
  { m.set(r, c, v) };                 // Method to set values by row/column
};

/// The VECTOR concept: a minimal description of what we expect from anything
/// that purports to be a vector (in the linear algebra sense, not in the
/// programming languages sense)
template <template <typename> typename M, typename T>
concept VECTOR = //
    requires(M<T> m, T v, uint64_t i) {
      { m.size };                      // A static field named `size`
      { m.get(i) } -> std::same_as<T>; // Method to get values by index
      { m.set(i, v) };                 // Method to set values by index
    };