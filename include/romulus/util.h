#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

inline std::string id_to_dns_name(uint64_t id) {
  return std::string("node") + std::to_string(id);
}
