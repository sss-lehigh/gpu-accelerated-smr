#include "concepts.h"
#include "logging.h"
#include "workload.h"
#include "state.h"

int main([[maybe_unused]]int argc, [[maybe_unused]]char* argv[]) {
  LOGGING_INFO("Starting SMR experiments...");

  WorkloadGenerator wg;
  auto ops = wg.generate(1000);

  uint64_t num_mats = 10;
  std::vector<DenseMat<int>> state_mats;
  state_mats.reserve(num_mats);
  for(uint64_t i = 0; i < num_mats; ++i) {
    state_mats.push_back(wg.generateMatrix());
  }

  State<int> st(state_mats);

  wg.print(0, 10);
  return 0;
}