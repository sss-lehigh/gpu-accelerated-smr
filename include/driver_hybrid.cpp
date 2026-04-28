#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <future>

#include "dag.h"       // Contains ExecutionGraph, ExecMode, and ExecTarget
#include "cpu/cpu.h"   // CpuExecutor
#include "gpu.cuh"     // GpuExecutor
#include "state.h"
#include "workload.h"  

int main(int argc, char* argv[]) {
  try {
    std::cout << "[1/5] Initializing Workload and DAG (HYBRID MODE)..." << std::endl;
    
    // 1. Initialize the ExecutionGraph in HYBRID mode
    ExecutionGraph graph(ExecMode::HYBRID);
    
    State<float> initstate(kNumMatrices);
    initstate.populate_random_state_matrix(1.0f, 100.0f);

    WorkloadGenerator wg;
    std::vector<op> ops = wg.generate(100, 5); 

    // 2. Ingest batch and calculate routing scores
    graph.ingest_batch(ops);
    const auto& dag = graph.get_dag();
    
    std::cout << "Done. Original Ops: " << ops.size() 
              << " | Fused Nodes in DAG: " << dag.size() << std::endl;

    std::cout << "[2/5] Generating parallel dispatch levels..." << std::endl;
    auto levels = graph.generate_levels();

    // Print routing decisions for debugging
    int cpu_count = 0, gpu_count = 0;
    for (const auto& [id, node] : dag) {
        if (node.target == ExecTarget::CPU) cpu_count++;
        else gpu_count++;
    }
    std::cout << "Routing Complete -> CPU Tasks: " << cpu_count 
              << " | GPU Tasks: " << gpu_count << std::endl;

    std::cout << "[3/5] Initializing Heterogeneous Executors..." << std::endl;
    CpuExecutor cpu_exec(ROWS, COLS);
    GpuExecutor gpu_exec(ROWS, COLS);
    
    std::atomic<int> op_counter{0};

    // Load initial state into both executors
    cpu_exec.load_state(initstate);
    gpu_exec.load_state(initstate);

    // Prepare GPU memory (only allocates for ExecTarget::GPU nodes)
    gpu_exec.prepare_dag(dag);

    std::cout << "\n[4/5] Launching Concurrent Hybrid Execution..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 3. LAUNCH CPU THREAD ASYNCHRONOUSLY
    // This allows the CPU to process its designated tasks while the main 
    // thread immediately moves on to dispatch the GPU tasks.
    auto cpu_future = std::async(std::launch::async, [&]() {
        cpu_exec.run(dag, levels, &op_counter);
    });

    // 4. LAUNCH GPU EXECUTION ON MAIN THREAD
    gpu_exec.run(dag, levels, &op_counter);

    // 5. SYNCHRONIZE
    // Wait for the CPU thread to finish its portion of the work
    cpu_future.get();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_time = end_time - start_time;
    
    int total_ops = op_counter.load();
    std::cout << "[5/5] Execution Complete.\n";
    
    // 6. Print true throughput metrics
    std::cout << "\n========================================" << std::endl;
    std::cout << "Total Operations (Including Fused): " << total_ops << std::endl;
    std::cout << "Hybrid Execution Time: " << total_time.count() << " ms" << std::endl;
    std::cout << "Hybrid Throughput: " << (total_ops / (total_time.count() / 1000.0)) << " ops/sec" << std::endl;
    std::cout << "========================================" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}