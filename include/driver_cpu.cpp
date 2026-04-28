#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>

#include "dag.h"       // Contains ExecutionGraph, ExecMode, and ExecTarget
#include "cpu/cpu.h" 
#include "state.h"
#include "workload.h"  // For WorkloadGenerator and ROWS/COLS

int main(int argc, char* argv[]) {
  try {
    std::cout << "[1/5] Initializing Workload and DAG..." << std::endl;
    
    // 1. Initialize the new ExecutionGraph in CPU-only benchmarking mode
    ExecutionGraph graph(ExecMode::BASELINE_CPU);
    
    State<float> initstate(kNumMatrices);
    initstate.populate_random_state_matrix(1.0f, 100.0f);

    WorkloadGenerator wg;
    std::vector<op> ops = wg.generate(100, 5); 

    // 2. Use the unified ingest_batch function
    graph.ingest_batch(ops);
    const auto& dag = graph.get_dag();
    
    std::cout << "Done. Original Ops: " << ops.size() 
              << " | Fused Nodes in DAG: " << dag.size() << std::endl;

    std::cout << "[2/5] Sorting DAG into parallel levels..." << std::endl;
    // 3. Generate levels directly from the graph
    auto levels = graph.generate_levels();
    
    // Print schedule (Replacing the deprecated Scheduler::print)
    for (size_t i = 0; i < levels.size(); ++i) {
        std::cout << "Level " << i << ": ";
        for (uint64_t id : levels[i]) {
            std::cout << id << " ";
        }
        std::cout << "\n";
    }

    std::cout << "[3/5] Initializing CPU Executor and Workspaces..." << std::endl;
    CpuExecutor executor(ROWS, COLS);
    
    // 4. Initialize the atomic operation counter
    std::atomic<int> op_counter{0};

    // BENCHMARK 1: SEQUENTIAL EXECUTION
    std::cout << "\n[4/5] Running Sequential Baseline..." << std::endl;
    executor.load_state(initstate); // Load fresh state matrices

    auto start_seq = std::chrono::high_resolution_clock::now();
    
    // Pass the counter pointer
    executor.run_sequential(dag, &op_counter);
    
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> seq_time = end_seq - start_seq;
    
    int seq_ops = op_counter.load();
    std::cout << "--> Sequential Time: " << seq_time.count() << " ms\n";
    std::cout << "--> Operations Processed (Including Fused): " << seq_ops << "\n";

    // BENCHMARK 2: PARALLEL DAG EXECUTION
    std::cout << "\n[5/5] Running Parallel DAG Execution..." << std::endl;
    executor.load_state(initstate); // Reset state matrices for a fair run

    // 5. CRITICAL: Reset the counter before the second benchmark
    op_counter.store(0); 

    auto start_par = std::chrono::high_resolution_clock::now();
    
    // Pass the counter pointer
    executor.run(dag, levels, &op_counter);
    
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> par_time = end_par - start_par;
    
    int par_ops = op_counter.load();
    std::cout << "--> Parallel DAG Time: " << par_time.count() << " ms\n";
    std::cout << "--> Operations Processed (Including Fused): " << par_ops << "\n";

    // 6. Print true throughput metrics
    std::cout << "\n========================================" << std::endl;
    std::cout << "CPU Speedup Factor: " << seq_time.count() / par_time.count() << "x" << std::endl;
    std::cout << "Seq Throughput: " << (seq_ops / (seq_time.count() / 1000.0)) << " ops/sec" << std::endl;
    std::cout << "Par Throughput: " << (par_ops / (par_time.count() / 1000.0)) << " ops/sec" << std::endl;
    std::cout << "========================================" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}