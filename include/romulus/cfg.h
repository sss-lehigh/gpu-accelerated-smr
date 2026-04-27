#pragma once

#include "cli.h"

namespace romulus {
constexpr const char* NODE_ID = "--node-id";
constexpr const char* REMOTES = "--remotes";
constexpr const char* REGISTRY_IP = "--registry-ip";
constexpr const char* TRANSPORT_TYPE = "--transport-type";
constexpr const char* DEV_NAME = "--dev-name";
constexpr const char* DEV_PORT = "--dev-port";
constexpr const char* NUM_QP = "--num-qp";
constexpr const char* POLICY = "--policy";
constexpr const char* OUTPUT_FILE = "--output-file";
constexpr const char* HELP = "--help";

inline auto ARGS = {
    U64_ARG(NODE_ID, "A numerical identifier for this node."),
    STR_ARG(REMOTES, "A comma-separated list of remote node names (Cloudlab DNS)."),
    STR_ARG_OPT(REGISTRY_IP,
                "IP address of machine hosting Memcached instance.",
                "10.10.1.1"),
    ENUM_ARG_OPT(TRANSPORT_TYPE, "RDMA transport type.", "RoCE", {"IB", "RoCE"}),
    STR_ARG_OPT(DEV_NAME, "Name of the RDMA device.", "mlx5_0"), // mlx5_0
    U64_ARG_OPT(DEV_PORT, "Device port.", 1),
    U64_ARG_OPT(NUM_QP, "Number of QP's to use.", 1),
    ENUM_ARG_OPT(POLICY, "Policy for how threads select QP's", "OTO", {"OTO", "RR", "RAND"}),
    STR_ARG_OPT(OUTPUT_FILE, "Name of the file to store experimental results.", "stats.csv"),
    BOOL_ARG_OPT(HELP, "Print this help message")};
}  // namespace romulus
