#pragma once

#include <infiniband/verbs.h>
#include <libmemcached/memcached.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "romulus/logging.h"

// ---------- Logging Macros ----------

#define XSTR(x) STR(x)
#define STR(x) #x
#define SOFTWARE_BARRIER "asm volatile(\"\": : :\" memory \")"
#define CACHE_PREFETCH_SIZE 64

// ---------- Romulus Namespace ----------

namespace romulus {
// --------- RAII Deleters ---------

struct MemcachedStDeleter {
  void operator()(struct memcached_st *st) {
    ROMULUS_DEBUG("Freeing memcached client: {:x}",
                  reinterpret_cast<uintptr_t>(st));
    memcached_free(st);
  }
};
using MemcachedStUniquePtr = std::unique_ptr<memcached_st, MemcachedStDeleter>;

struct QpDeleter {
  void operator()(struct ibv_qp *qp) {
    ROMULUS_DEBUG("Destroying queue pair: {}", reinterpret_cast<uintptr_t>(qp));
    if (ibv_destroy_qp(qp) != 0) {
      ROMULUS_FATAL("Failed to destroy QP");
    }
  }
};
using QpUniquePtr = std::unique_ptr<ibv_qp, QpDeleter>;

struct CqDeleter {
  void operator()(ibv_cq *cq) {
    if (cq)
      ibv_destroy_cq(cq);
  }
};
using CqUniquePtr = std::unique_ptr<ibv_cq, CqDeleter>;

struct ContextDeleter {
  void operator()(struct ibv_context *context) {
    ROMULUS_DEBUG("Closing device: {}", context->device->dev_name);
    ROMULUS_ASSERT(ibv_close_device(context) == 0, "Failed to close device: {}",
                   context->device->dev_name);
  }
};
using ContextUniquePtr = std::unique_ptr<ibv_context, ContextDeleter>;

struct PdDeleter {
  void operator()([[maybe_unused]] struct ibv_pd *pd) {
    ROMULUS_DEBUG("Deallocating protection domain: {}",
                  reinterpret_cast<uintptr_t>(pd));
    if (ibv_dealloc_pd(pd) != 0) {
      ROMULUS_FATAL("Failed to deallocate pd: {}",
                    reinterpret_cast<uintptr_t>(pd));
    }
  }
};
using PdUniquePtr = std::unique_ptr<ibv_pd, PdDeleter>;

struct MrDeleter {
  void operator()(struct ibv_mr *mr) {
    ROMULUS_DEBUG("Deregistering memory region: {}",
                  reinterpret_cast<uintptr_t>(mr));
    ROMULUS_ASSERT(ibv_dereg_mr(mr) == 0, "Failed to deregister memory region");
  }
};
using MrUniquePtr = std::unique_ptr<ibv_mr, MrDeleter>;

// --------- Address Structures ---------

constexpr char kRemoteAddrDelim = ':';

struct ConnInfo {
  uint32_t qp_num; // QP number
  uint16_t lid;    // LID of the IB port
  uint8_t gid[16]; // Global id

  std::string ToString() const {
    std::stringstream ss;
    ss << std::dec << qp_num << kRemoteAddrDelim;
    ss << std::dec << lid << kRemoteAddrDelim;
    for (int i = 0; i < 16; ++i) {
      ss << std::dec << static_cast<uint32_t>(gid[i]);
      if (i < 15)
        ss << kRemoteAddrDelim;
    }
    return ss.str();
  }

  void FromString(std::string str) {
    std::replace(str.begin(), str.end(), kRemoteAddrDelim, ' ');
    std::stringstream ss(str);
    ss >> std::dec >> qp_num;
    ss >> std::dec >> lid;
    uint32_t byte;
    for (int i = 0; i < 16; ++i) {
      ss >> std::dec >> byte;
      gid[i] = static_cast<uint8_t>(byte);
    }
  }
};

struct AddrInfo {
  uint64_t addr;   // Buffer address
  uint32_t offset; // Buffer offset
  uint32_t length; // Buffer size
  uint32_t key;    // Access key

  std::string ToString() const {
    std::stringstream ss;
    ss << std::hex << addr << kRemoteAddrDelim;
    ss << std::dec << offset << kRemoteAddrDelim;
    ss << std::dec << length << kRemoteAddrDelim;
    ss << std::dec << key;
    return ss.str();
  }

  void FromString(std::string str) {
    std::replace(str.begin(), str.end(), kRemoteAddrDelim, ' ');
    std::stringstream ss(str);
    ss >> std::hex >> addr;
    ss >> std::dec >> offset;
    ss >> std::dec >> length;
    ss >> std::dec >> key;
  }
};

using LocalAddr = AddrInfo;

class RemoteAddr {
public:
  ConnInfo conn_info;
  AddrInfo addr_info;

  void SetConnInfo(const struct ConnInfo &conn) {
    conn_info.qp_num = conn.qp_num;
    conn_info.lid = conn.lid;
    *reinterpret_cast<uint64_t *>(&conn_info.gid[0]) =
        *reinterpret_cast<const uint64_t *>(&conn.gid[0]);
    *reinterpret_cast<uint64_t *>(&conn_info.gid[8]) =
        *reinterpret_cast<const uint64_t *>(&conn.gid[8]);
  }

  void SetAddrInfo(const struct AddrInfo &info) {
    addr_info.addr = info.addr;
    addr_info.length = info.length;
    addr_info.key = info.key;
  }

  inline bool operator==(const RemoteAddr &rhs) const {
    return (rhs.conn_info.qp_num == conn_info.qp_num &&
            rhs.conn_info.lid == conn_info.lid &&
            *(reinterpret_cast<const uint64_t *>(&rhs.conn_info.gid[0])) ==
                *(reinterpret_cast<const uint64_t *>(&conn_info.gid[0])) &&
            *(reinterpret_cast<const uint64_t *>(&rhs.conn_info.gid[8])) ==
                *(reinterpret_cast<const uint64_t *>(&conn_info.gid[8])) &&
            rhs.addr_info.addr == addr_info.addr &&
            rhs.addr_info.length == addr_info.length &&
            rhs.addr_info.key == addr_info.key);
  }
};

// --------- Helpers ---------

inline uint32_t GetQuorum(uint32_t system_size) {
  return (system_size / 2) + 1;
}

} // namespace romulus
