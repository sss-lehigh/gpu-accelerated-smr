#include <csignal>
#include <filesystem>
#include <functional>
#include <random>
#include <string>

#include "cfg.h"
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

#include "mu/mu_impl.h"
#include "dag.h"


#define PAXOS_NS paxos_st
constexpr uint32_t kNumProposals = 8092;
constexpr uint32_t kMaxBufSize = 1024;

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

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(0, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

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
  ROMULUS_INFO("!> [CONF] policy={}", policy);
  ROMULUS_INFO("!> [CONF] duration={}_ms", duration.count());
  ROMULUS_INFO("!> [CONF] system_size={}", system_size);
  ROMULUS_INFO("!> [CONF] output file={}", output_file);

  INIT_CONSENSUS(transport_flag, buf_size, mach_map);
  FILL_PROPOSALS();

  std::function<void(void)> init = SYNC_NODES;
  std::function<void(void)> exec = EXEC_LATENCY;
  std::function<void(void)> done = DONE_LATENCY;
  std::function<void(std::ofstream&)> calc = CALC_LATENCY;
  std::function<void(void)> reset = RESET;

  init();

  ROMULUS_INFO("Starting latency test");
  
  auto testtime_us =
      std::chrono::duration_cast<std::chrono::microseconds>(testtime);
  ROMULUS_STOPWATCH_BEGIN();
  // size_t iterations = 0;
  size_t last_offload_idx = 0;
  DagGenerator dag_generator;

  while (ROMULUS_STOPWATCH_RUNTIME(ROMULUS_MICROSECONDS) <
         static_cast<uint64_t>(testtime_us.count())) {
    for (uint32_t i = 0; i < loop; ++i) {
      // Fixed leader node0
      if (id == 0) {
        if(i >= kMaxBufSize){
          // take slice of last_offload_idx to last_offload_idx + kMaxBufSize
          auto slice = std::vector<op>(
              ops.begin() + last_offload_idx,
              ops.begin() + std::min(last_offload_idx + kMaxBufSize, (size_t)ops.size()));
          ROMULUS_INFO("Consensus buffer is full. Triggering offloading process...");

          // dag_generator.build_dag(slice);
        }
        exec();
        busy_wait(sleep);
      }
    }
  }

  init();  // sync

  ROMULUS_INFO("Experiment is finished. Cleaning up...");

  done();  // cleanup

  if (id == (int)system_size - 1) {
    calc(outfile);
    calc = CALC_THROUGHPUT;
    calc(outfile);
  }

  outfile.close();

  for (auto& p : proposals) {
    delete[] p.second;
  }

  return 0;
}