#pragma once

// ROMULUS = Research Oriented MEasurements
// Provides macros for counters and cycle-accurate timing.

#include <cinttypes>
#include <mutex>
#include <string>
#include <unordered_map>

// --------------------- Timing constants ---------------------
#define ROMULUS_SECONDS 1000000
#define ROMULUS_MILLISECONDS 1000
#define ROMULUS_MICROSECONDS 1
#define ROMULUS_NANOSECONDS 0.0001

#define ROMULUS_TSC_FREQ_MHZ 4500ULL  // Adjust per machine

// --------------------- Stopwatch macros ---------------------
#define ROMULUS_STOPWATCH_DECLARE()                              \
  uint32_t romulus_t1_high, romulus_t1_low, romulus_t2_high, romulus_t2_low; \
  [[maybe_unused]] uint64_t romulus_begin, romulus_end, romulus_t1, romulus_t2

#define ROMULUS_STOPWATCH_BEGIN()                                           \
  asm volatile(                                                             \
      "MFENCE\n\t"                                                          \
      "RDTSCP\n\t"                                                          \
      : "=a"(romulus_t1_low), "=d"(romulus_t1_high)::"%ecx");                     \
  romulus_begin = ((static_cast<uint64_t>(romulus_t1_high) << 32) + romulus_t1_low); \
  romulus_end = romulus_begin

#define ROMULUS_STOPWATCH_RUNTIME(scaling_factor)                           \
  [&]() {                                                                   \
    asm volatile(                                                           \
        "RDTSCP\n\t"                                                        \
        "LFENCE\n\t"                                                        \
        : "=a"(romulus_t2_low), "=d"(romulus_t2_high)::"%ecx");                   \
    romulus_end = ((static_cast<uint64_t>(romulus_t2_high) << 32) + romulus_t2_low); \
    return ((static_cast<double>(romulus_end - romulus_begin) /                   \
             static_cast<double>(ROMULUS_TSC_FREQ_MHZ * scaling_factor)));  \
  }()

#define ROMULUS_STOPWATCH_START()                       \
  asm volatile(                                         \
      "MFENCE\n\t"                                      \
      "RDTSCP\n\t"                                      \
      : "=a"(romulus_t1_low), "=d"(romulus_t1_high)::"%ecx"); \
  romulus_t1 = ((static_cast<uint64_t>(romulus_t1_high) << 32) + romulus_t1_low)

#define ROMULUS_STOPWATCH_SPLIT(scaling_factor)                            \
  [&]() {                                                                  \
    asm volatile(                                                          \
        "RDTSCP\n\t"                                                       \
        "LFENCE\n\t"                                                       \
        : "=a"(romulus_t2_low), "=d"(romulus_t2_high)::"%ecx");                  \
    romulus_t2 = ((static_cast<uint64_t>(romulus_t2_high) << 32) + romulus_t2_low); \
    return ((static_cast<double>(romulus_t2 - romulus_t1) /                      \
             static_cast<double>(ROMULUS_TSC_FREQ_MHZ * scaling_factor))); \
  }()

// --------------------- Counters ---------------------
#ifndef NROMULUS

inline std::mutex romulus_mutex;
inline std::unordered_map<std::string, uint64_t> romulus_global_counters;
inline thread_local std::unordered_map<std::string, uint64_t>
    romulus_thread_counters;

#define ROMULUS_COUNTER(c) romulus_global_counters[c] = 0
#define ROMULUS_COUNTER_INC(c) romulus_thread_counters[c] += 1
#define ROMULUS_COUNTER_DEC(c) romulus_thread_counters[c] -= 1
#define ROMULUS_COUNTER_ACC(c)                          \
  {                                                     \
    std::lock_guard<std::mutex> lock(romulus_mutex);       \
    romulus_global_counters[c] += romulus_thread_counters[c]; \
  }
#define ROMULUS_COUNTER_GET(c) romulus_global_counters[c]

#else

#define ROMULUS_COUNTER(c) ((void)0)
#define ROMULUS_COUNTER_INC(c) ((void)0)
#define ROMULUS_COUNTER_DEC(c) ((void)0)
#define ROMULUS_COUNTER_ACC(c) ((void)0)
#define ROMULUS_COUNTER_GET(c) 0

#endif
