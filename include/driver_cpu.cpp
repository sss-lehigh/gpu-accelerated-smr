#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "dag.h"
#include "scheduler.h"
#include "cpu/cpu.h" 
#include "state.h"

int main(int argc, char* argv[]) {
  try {
    std::cout << "[1/5] Initializing DAG Generator..." << std::endl;
    DagGenerator builder;
    
    State<float> initstate;
    initstate.populate_random_state_matrix(1.0f, 100.0f);

    WorkloadGenerator wg;
    std::vector<op> ops = wg.generate(100, 5); 

    builder.build_dag(ops);
    auto& dag = builder.get_dag();
    std::cout << "Done. Fused Nodes in DAG: " << dag.size() << std::endl;

    std::cout << "[2/5] Sorting DAG into parallel levels..." << std::endl;
    auto levels = Scheduler::get_levels(dag);
    Scheduler::print(levels); 

    std::cout << "[3/5] Initializing CPU Executor and Workspaces..." << std::endl;
    CpuExecutor executor(ROWS, COLS);
    
    // BENCHMARK 1: SEQUENTIAL EXECUTION
    std::cout << "\n[4/5] Running Sequential Baseline..." << std::endl;
    executor.load_state(initstate); // Load fresh state matrices

    auto start_seq = std::chrono::high_resolution_clock::now();
    
    executor.run_sequential(dag);
    
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> seq_time = end_seq - start_seq;
    std::cout << "--> Sequential Time: " << seq_time.count() << " ms" << std::endl;

    // BENCHMARK 2: PARALLEL DAG EXECUTION
    std::cout << "\n[5/5] Running Parallel DAG Execution..." << std::endl;
    executor.load_state(initstate); // Reset state matrices for a fair run

    auto start_par = std::chrono::high_resolution_clock::now();
    
    executor.run(dag, levels);
    
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> par_time = end_par - start_par;
    std::cout << "--> Parallel DAG Time: " << par_time.count() << " ms" << std::endl;

    // RESULTS
    std::cout << "\n========================================" << std::endl;
    std::cout << "CPU Speedup Factor: " << seq_time.count() / par_time.count() << "x" << std::endl;
    std::cout << "========================================" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}