#include "concepts.h"
#include "logging.h"
#include "workload.h"

int main([[maybe_unused]]int argc, [[maybe_unused]]char* argv[]) {
  LOGGING_INFO("Starting SMR experiments...");

  WorkloadGenerator wg;
  wg.generate(1000);

  wg.print(0, 10);
  return 0;
}