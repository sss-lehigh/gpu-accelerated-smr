#pragma once

#include <random>

enum POLICY : uint32_t {
  // One to one relation between qp's and threads
  OTO = 0,
  // Roundrobin-style iteration over the qp's to assign to threads
  RR = 1,
  // Random assignment of qp's to threads
  RAND = 2
};

class QpPolicy {
 public:
  QpPolicy(uint32_t num_qps) : policy_("RR"), num_qps_(num_qps) {}
  QpPolicy(uint32_t num_qps, std::string policy, uint32_t tid)
      : policy_(policy), num_qps_(num_qps), thread_id_(tid) {}
  uint32_t get_idx() {
    if (policy_ == "RR") {
      return thread_id_ % num_qps_;
    } else if (policy_ == "RAND") {
      thread_local std::mt19937 generator(std::random_device{}());
      std::uniform_int_distribution<uint32_t> distribution(0, num_qps_);
      return distribution(generator);
    } else {
      return thread_id_;
    }
  }

 private:
  std::string policy_;
  uint32_t num_qps_;
  // uint32_t counter_;
  uint32_t thread_id_;
};
