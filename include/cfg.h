#pragma once

#include <romulus/cli.h>

namespace romulus {
constexpr const char* HOSTNAME = "--hostname";
constexpr const char* TESTTIME = "--testtime";
constexpr const char* LOOP = "--loop";
constexpr const char* CAPACITY = "--capacity";
constexpr const char* BUF_SIZE = "--buf-size";
constexpr const char* SLEEP = "--sleep";
constexpr const char* STABLE_LEADER = "--stable-leader";
constexpr const char* GPU_ENABLED = "--gpu-enabled";
constexpr const char* MODE = "--mode";
constexpr const char* NUM_STATE_MAT = "--num-state-mat";
constexpr const char* MAT_SIZE = "--mat-size";


// Cloudlab notes:
// r320
// - device: mlx4_0
// - port 1: IB
// - port 2: RoCE

// xl170
// - device: mlx5_0 (10 Gbps)
//  - port 1: RoCE
// - device: mlx5_3 (25 Gbps)
//  - port 1: RoCE

inline auto EXTRA_ARGS = {
    STR_ARG(HOSTNAME, "Hostname of this node."),
    U64_ARG_OPT(TESTTIME, "Experiment duration in seconds", 30),
    U64_ARG_OPT(LOOP, "Number of iterations between runtime checks.", 1000),
    U64_ARG_OPT(CAPACITY, "Capacity of the replicated log.", (1ULL << 12)),
    U64_ARG_OPT(BUF_SIZE, "Buffer size for remote writes.", 64),
    U64_ARG_OPT(SLEEP, "Sleep interval between proposals in ms", 10),
    BOOL_ARG_OPT(STABLE_LEADER,
                 "If true, only a single node proposes commands."),
    BOOL_ARG_OPT(GPU_ENABLED, "If true, GPU groups are enabled."),
    ENUM_ARG_OPT(MODE, "Execution mode (SERIAL | DAG)", "DAG", {"SERIAL", "DAG"}),
    U64_ARG_OPT(NUM_STATE_MAT, "Number of state matrices.", 5),
    U64_ARG_OPT(MAT_SIZE, "Size of each matrix.", 512)};
};  // namespace romulus