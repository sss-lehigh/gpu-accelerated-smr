#include <barrier>
#include <csignal>
#include <filesystem>
#include <functional>
#include <iostream>
#include <random>
#include <string>

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

#define PAXOS_NS paxos_st
constexpr uint32_t kNumProposals = 8092;

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
  std::function<void(std::ofstream&)> calc = CALC_LATENCY;
  std::function<void(void)> reset = RESET;

  init();

  // Initializing every components
  // WorkloadGenerator wg;
  DagGenerator builder;

  // Initialize the State Machine exactly once
  State<float> initstate;
  initstate.populate_random_state_matrix(1.0f, 100.0f);

  CpuExecutor cpu_exec(ROWS, COLS);
  GpuExecutor gpu_exec(ROWS, COLS);

  cpu_exec.load_state(initstate);
  gpu_exec.load_state(initstate);

  std::atomic<bool> handler_running = true;
  std::barrier commit_barrier(2);
  ROMULUS_INFO("Initialization is finished. Launching commit thread...");

  // Track how far into the master 'ops' array we have committed
  size_t current_commit_idx = 0;

  auto commit_handler = std::thread([&]() {
    PinToCore(1);
    std::atomic<int> op_counter = 0;
    while (handler_running.load(std::memory_order_relaxed) == true) {
      // Wait for the main thread to signal that a batch of proposals has been
      // sent
      commit_barrier.arrive_and_wait();
      // Need to do a quick check after all that wait time
      if (handler_running.load(std::memory_order_relaxed) == false) {
        break;
      }

      // Only the leader is allowed to process the DAG
      if (id == 0) {
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

        // Reset the DAG Generator to clear memory and dependencies from the
        // previous batch
        builder.reset();

        // Build the new DAG
        builder.build_dag(current_batch_ops);
        auto& dag = builder.get_dag();
        // [Rishad] : At this point, we have the DAG built. Why do we have to
        // rely on the scheduler to give us a score? Can't we just define a
        // utility function that takes a couple of factors {e.g. num levels, num
        // ops per level, num total ops} and gives us a score that we can use to
        // decide whether to run on cpu or gpu? Can't we do that in this scope?
        auto levels = Scheduler::get_levels(dag);
        bool is_serial = (mode == "SERIAL");
        // Execute the Dynamic Batch
        if (cpu_enabled && is_serial) {
          ROMULUS_INFO("[Commit handler] Running on CPU in SERIAL mode");
          cpu_exec.run_sequential(dag, &op_counter);
        }
        if (cpu_enabled && !is_serial) {
          ROMULUS_INFO("[Commit handler] Running on CPU in DAG mode");
          cpu_exec.run(dag, levels, &op_counter);
        }
        // at this point it must be a gpu execution
        gpu_exec.prepare_dag(dag);
        if (gpu_enabled && is_serial) {
          ROMULUS_INFO("[Commit handler] Running on GPU in SERIAL mode");
          gpu_exec.run_sequential(dag, &op_counter);
        }
        if (gpu_enabled && !is_serial) {
          ROMULUS_INFO("[Commit handler] Running on GPU in DAG mode");
          gpu_exec.run(dag, levels, &op_counter);
        }

        if (is_serial) {
          ROMULUS_INFO("[Commit handler] Running on GPU in SERIAL mode");
          gpu_exec.run_sequential(dag, &op_counter);
        } else {
          ROMULUS_INFO("[Commit handler] Running on GPU in DAG mode");
          gpu_exec.run(dag, levels, &op_counter);
        }
        // Ensure GPU is finished before returning to the next barrier
        cudaDeviceSynchronize();
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
  // size_t iterations = 0; // turning off iterations variable for now (causing
  // -werror)
  while (ROMULUS_STOPWATCH_RUNTIME(ROMULUS_MICROSECONDS) <
         static_cast<uint64_t>(testtime_us.count())) {
    for (uint32_t i = 0; i < loop; ++i) {
      // Fixed leader node0
      if (id == 0) {
        if (fuo >= buf_size) {
          ROMULUS_INFO("Flushing buffer ({} bytes) to commit handler...",
                       fuo * sizeof(op));
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

  done();  // cleanup

  if (id == (int)system_size - 1) {
    std::ofstream outfile(output_file);
    calc(outfile);
    calc = CALC_THROUGHPUT;
    calc(outfile);
    outfile.close();
  }

  for (auto& p : proposals) {
    delete[] p.second;
  }

  return 0;
}