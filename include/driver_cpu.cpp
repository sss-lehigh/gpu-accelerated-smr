#include <iostream>
#include <string>
#include <vector>

#include "dag.h"
#include "scheduler.h"
#include "cpu/cpu.h" 
#include "state.h"

int main(int argc, char* argv[]) {
  try {
    std::cout << "[1/4] Initializing DAG Generator..." << std::endl;
    DagGenerator builder;
    
    // Use the same State class we fixed earlier
    State<float> initstate;
    initstate.populate_random_state_matrix(1.0f, 100.0f);

    // Generate or load your workload
    WorkloadGenerator wg;
    std::vector<op> ops = wg.generate(100, 5); 

    builder.build_dag(ops);
    auto& dag = builder.get_dag();
    std::cout << "Done. Nodes in DAG: " << dag.size() << std::endl;

    std::cout << "[2/4] Sorting DAG into parallel levels..." << std::endl;
    auto levels = Scheduler::get_levels(dag);
    Scheduler::print(levels); 

    std::cout << "[3/4] Initializing CPU Executor and Workspaces..." << std::endl;
    // ROWS and COLS should be defined in your common headers
    CpuExecutor executor(ROWS, COLS);
    
    // Copy the initial state into the executor's host memory
    executor.load_state(initstate); 

    std::cout << "[4/4] Executing on CPU (Multi-threaded OpenMP)..." << std::endl;
    // The executor will handle nested parallelism internally
    executor.run(dag, levels);

    std::cout << "SUCCESS: CPU Workload completed." << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}