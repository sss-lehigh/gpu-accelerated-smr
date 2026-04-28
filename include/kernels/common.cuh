#pragma once
#include <iostream>

// Standard macro for catching CUDA errors
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__             \
                << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" \
                << std::endl;                                                  \
      __builtin_trap();                                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
