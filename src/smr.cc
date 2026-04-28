#include <barrier>
#include <csignal>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <cuda_runtime.h>

#include "cfg.h"
#include "cpu/cpu.h"
#include "dag.h"
#include "gpu.cuh"
#include "mu/mu_impl.h"
#include "romulus/cfg.h"
#include "romulus/common.h"
#include "romulus/connection_manager.h"
#include "romulus/device.h"
#include "romulus/memblock.h"
#include "romulus/qp_pol.h"
#include "romulus/stats.h"
#include "romulus/util.h"
#include "state.h"
#include "util.h"
#include "workload.h"

#define PAXOS_NS paxos_st

void signal_handler(int signum) {
  if (signum == SIGTSTP) {
    write(STDOUT_FILENO, "SIGINT caught\n", 14);
    dump_requested_.store(true, std::memory_order_relaxed);
  }
}

int main(int argc, char* argv[]) {
  struct sigaction sa{};
  sa.sa_handler = signal_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGTSTP, &sa, nullptr);

  PinToCore(0);

  ROMULUS_STOPWATCH_DECLARE();
  romulus::INIT();
  auto args = std::make_shared<romulus::ArgMap>();
  args->import(romulus::ARGS);
  args->import(romulus::EXTRA_ARGS);
  args->parse(argc, argv);
  INGEST_ARGS(args);

  // Print configuration.
  ROMULUS_INFO("Experimental Configuration:");
  ROMULUS_INFO("!> [CONF] hostname={}", hostname);
  ROMULUS_INFO("!> [CONF] host id={}", id);
  ROMULUS_INFO("!> [CONF] registry ip={}", registry_ip);
  ROMULUS_INFO("!> [CONF] output file={}", output_file);
  ROMULUS_INFO("!> [CONF] testtime={}_s", testtime.count());
  ROMULUS_INFO("!> [CONF] device name={}", dev_name);
  ROMULUS_INFO("!> [CONF] device port={}", dev_port);
  ROMULUS_INFO("!> [CONF] transport type={}", transport);
  ROMULUS_INFO("!> [CONF] loop={}", loop);
  ROMULUS_INFO("!> [CONF] capacity={}", capacity);
  ROMULUS_INFO("!> [CONF] buf_size={}", buf_size);
  ROMULUS_INFO("!> [CONF] sleep={}_ms", sleep.count());
  ROMULUS_INFO("!> [CONF] leader_fixed={}", leader_fixed);
  // ROMULUS_INFO("!> [CONF] policy={}", policy);
  // ROMULUS_INFO("!> [CONF] duration={}_ms", duration.count());
  ROMULUS_INFO("!> [CONF] system_size={}", system_size);
  ROMULUS_INFO("!> [CONF] cpu_enabled={}", cpu_enabled);
  ROMULUS_INFO("!> [CONF] gpu_enabled={}", gpu_enabled);
  ROMULUS_INFO("!> [CONF] mode={}", mode);
  INIT_CONSENSUS(transport_flag, buf_size, mach_map);
  FILL_PROPOSALS();

  std::function<void(void)> init = SYNC_NODES;
  std::function<void(void)> exec = EXEC_LATENCY;
  std::function<void(void)> done = DONE_LATENCY;
  std::function<void(std::tuple<double, double, double, double>* result)> calc = CALC_LATENCY;
  std::function<void(void)> reset = RESET;

  init();

  // Map config flags to the new ExecMode
  ExecMode e_mode = ExecMode::HYBRID;
  if (cpu_enabled && !gpu_enabled) e_mode = ExecMode::BASELINE_CPU;
  if (!cpu_enabled && gpu_enabled) e_mode = ExecMode::BASELINE_GPU;

  // Initialize the combined ExecutionGraph
  ExecutionGraph graph(e_mode);

  // Initialize the State Machine exactly once
  State<float> initstate(num_state_mat, mat_size);
  initstate.populate_random_state_matrix(1.0f, 100.0f);

  // CUDA Unified Memory allocation
  float** unified_state_matrices = new float*[num_state_mat];
  size_t matrix_bytes = mat_size * mat_size * sizeof(float);

  for (int i = 0; i < num_state_mat; ++i) {
    cudaMallocManaged(&unified_state_matrices[i], matrix_bytes);
  }

  // Pass the exact same memory pointers to both executors
  CpuExecutor cpu_exec(mat_size, num_state_mat, unified_state_matrices);
  GpuExecutor gpu_exec(mat_size, num_state_mat, unified_state_matrices);

  // Load state only once, since both executors now point to the same physical memory
  gpu_exec.load_state(initstate);

  std::atomic<bool> handler_running = true;
  std::barrier commit_barrier(2);
  ROMULUS_INFO("Initialization is finished. Launching commit thread...");

  // Track how far into the master 'ops' array we have committed
  size_t current_commit_idx = 0;
  std::atomic<int> op_counter = 0;

  auto commit_handler = std::thread([&]() {
    PinToCore(1);
    while (handler_running.load(std::memory_order_relaxed) == true) {
      // Wait for the main thread to signal that a batch of proposals has been sent
      commit_barrier.arrive_and_wait();
      // Need to do a quick check after all that wait time
      if (handler_running.load(std::memory_order_relaxed) == false) {
        break;
      }

      // Only the leader is allowed to process the DAG
      if (id == 0) {
        auto start_time = std::chrono::high_resolution_clock::now();
        // Fetch the REAL batch corresponding to what MU just committed
        std::vector<op> current_batch_ops;
        current_batch_ops.reserve(buf_size);

        // Slice the next 'buf_size' operations from the master 'ops' array
        size_t batch_end = std::min(current_commit_idx + buf_size, ops.size());
        for (size_t i = current_commit_idx; i < batch_end; ++i) {
          current_batch_ops.push_back(ops[i]);
        }

        // Update the tracker for the next batch
        current_commit_idx = batch_end;

        // If we ran out of proposals in the master list, loop back around
        if (current_commit_idx >= ops.size()) {
          current_commit_idx = 0;
        }

        graph.reset();
        graph.ingest_batch(current_batch_ops);
        const auto& dag = graph.get_dag();
        auto levels = graph.generate_levels();

        bool is_serial = (mode == "SERIAL");

        // Execution Dispatch Logic
        if (e_mode == ExecMode::BASELINE_CPU) {
          if (is_serial)
            cpu_exec.run_sequential(dag, &op_counter);
          else
            cpu_exec.run(dag, levels, &op_counter);
        } else if (e_mode == ExecMode::BASELINE_GPU) {
          gpu_exec.prepare_dag(dag);
          if (is_serial)
            gpu_exec.run_sequential(dag, &op_counter);
          else
            gpu_exec.run(dag, levels, &op_counter);
        } else if (e_mode == ExecMode::HYBRID) {
          // In Hybrid mode, both devices need access to the DAG
          gpu_exec.prepare_dag(dag);

          if (is_serial) {
            // Strictly sequential hybrid execution (for debugging)
            // Executors will internally check node.target
            cpu_exec.run_sequential(dag, &op_counter);
            gpu_exec.run_sequential(dag, &op_counter);
          } else {
            // Concurrent hybrid execution
            // Launch the CPU executor asynchronously so it runs at the same time as the GPU
            auto cpu_future = std::async(std::launch::async, [&]() {
              cpu_exec.run(dag, levels, &op_counter);
            });

            // The main thread drives the GPU executor
            gpu_exec.run(dag, levels, &op_counter);

            // Wait for the CPU thread to finish its portion of the DAG
            cpu_future.get();
          }
        }

        // Ensure GPU is finished before returning to the next barrier
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto batch_latency =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                  start_time)
                .count();
        commit_latencies.emplace_back(batch_latency);
      }
    }
  });

  ROMULUS_INFO("Starting latency test");
  // size_t iterations = 0;
  // size_t last_offload_idx = 0; // turning off this variable to avoid -werror
  // DagGenerator dag_generator;
  uint32_t fuo = 0;

  std::cout << "Wait some time" << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(2 + system_size - id));

  auto testtime_us =
      std::chrono::duration_cast<std::chrono::microseconds>(testtime);
  ROMULUS_STOPWATCH_BEGIN();
  // size_t iterations = 0; // turning off iterations variable for now
  while (ROMULUS_STOPWATCH_RUNTIME(ROMULUS_MICROSECONDS) <
         static_cast<uint64_t>(testtime_us.count())) {
    for (uint32_t i = 0; i < loop; ++i) {
      // Fixed leader node0
      if (id == 0) {
        if (fuo >= buf_size) {
          // ROMULUS_INFO("Flushing buffer ({} bytes) to commit handler...",
          //              fuo * sizeof(op));
          // Signal the commit handler to process the batch of proposals
          commit_barrier.arrive_and_wait();
          // Reset offset for the next batch
          fuo = 0;
        }
        exec();
        ++fuo;
      }
    }
  }
  handler_running.store(false, std::memory_order_relaxed);
  commit_barrier
      .arrive_and_drop();  // Wake up the commit thread to exit cleanly
  commit_handler.join();
  init();  // sync

  ROMULUS_INFO("Experiment is finished. Cleaning up...");

  if (id == 0) {
    // Calculate End-to-End Batch Latency and Throughput
    std::tuple<double, double, double, double> cons_latency_result;
    std::tuple<double, double, double, double> commit_latency_result;
    calc(&cons_latency_result);
    calc(&commit_latency_result);
    double cons_lat_avg = std::get<0>(cons_latency_result);
    double commit_lat_avg = std::get<0>(commit_latency_result);
    double e2e_lat_avg = cons_lat_avg + commit_lat_avg;
                         // calculate goodput as op-tracker / total time
                         double seconds = testtime_us.count() / 1e6;
    double goodput =
        op_counter.load(std::memory_order_relaxed) / seconds / 1e6;  // in MOPS

    std::stringstream ss;
    ss << system_size << "," << mat_size << "," << buf_size << ","
       << num_state_mat << "," << cpu_enabled << "," << gpu_enabled << ","
       << mode << "," << cons_lat_avg << "," << e2e_lat_avg << "," << goodput << std::endl;
    ROMULUS_INFO("[PARSE] {}", ss.str());
  }

  for (auto& p : proposals) {
    delete[] p.second;
  }

  // Cleanup unified memory
  for (int i = 0; i < num_state_mat; ++i) {
    cudaFree(unified_state_matrices[i]);
  }
  delete[] unified_state_matrices;

  return 0;
}