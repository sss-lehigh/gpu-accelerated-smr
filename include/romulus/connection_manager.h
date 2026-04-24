#pragma once

#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "device.h"
#include "memblock.h"
#include "rc.h"
#include "registry.h"

#define NUM_LOOPBACK_QPS 3

struct Barrier {
  uint64_t counter;
  std::string ToString() const { return std::to_string(counter); }
  void FromString(const std::string& s) { counter = std::atoi(s.c_str()); }
};

namespace romulus {

const uint64_t kLoopback_1 = 0;
const uint64_t kLoopback_2 = std::numeric_limits<uint64_t>::max();
const uint64_t kLoopback_3 = std::numeric_limits<uint64_t>::max() - 1;

// The ConnectionManager is responsible for setting up and tearing down
// connections between nodes for a given MemBlock. It interacts with the
// registry in order to do so.
class ConnectionManager {
 public:
  ConnectionManager() = default;
  ConnectionManager(std::string_view host, ConnectionRegistry* registry,
                    uint64_t id, uint64_t system_size, uint64_t num_qps)
      : host_(host),
        registry_(registry),
        id_(id),
        system_size_(system_size),
        num_qps_(num_qps) {
    // Register the barrier key if we are coordinator
    barrier_key_ = MakeBarrierKey();
    if (id_ == 0) {
      registry_->Register<Barrier>(barrier_key_, Barrier());
    } else {
      sleep(3);
    }
  }

  bool Register(const Device& dev, const MemBlock& memblock) {
    // Created shared CQ if first call
    if (!shared_cq_) {
      auto* context = dev.GetContext();
      int cq_size = (num_qps_ * system_size_ + 2) * ROMULUS_RC_CQ_SIZE;
      shared_cq_.reset(ibv_create_cq(context, cq_size, nullptr, nullptr, 0));
    }

    auto info_map = memblock.GetAllAddrInfos(/*for_remote=*/true);
    auto pd = memblock.GetPd();

    for (int i = 0; i < NUM_LOOPBACK_QPS; ++i) {
      uint64_t loopback_id;
      switch(i) {
        case 0:
          loopback_id = kLoopback_1;
          break;
        case 1:
          loopback_id = kLoopback_2;
          break;
        case 2:
          loopback_id = kLoopback_3;
          break;
        default:
          ROMULUS_FATAL("Invalid loopback index: {}", i);
          return false;
      }
      // Create and a loopback connection for the MemBlock
      std::string key = MakeConnKey(id_, id_, loopback_id);
      auto conn_iter = connections_.emplace(
          key, i == 0 ? ReliableConnection(dev, shared_cq_.get()) : ReliableConnection(dev)); 
      (conn_iter.first->second).Init(pd);
      for (const auto& i : *info_map) {
        key = MakeAddrKey(id_, memblock.GetBlockId(), i.first);
        remote_addrs_[key].SetConnInfo(
            conn_iter.first->second.GetLocalPeerConnInfo());
        if (!memblock.MakeAddrInfoForRemoteOp(i.first, std::nullopt,
                                              std::nullopt,
                                              &remote_addrs_[key].addr_info)) {
          return false;
        }
        ROMULUS_DEBUG("Loopback {} registered locally {}", loopback_id, key);
      }
    }

    // Register remote addresses for remote peers
    // We reserve qp_id 0 for loopback

    // NB: For the purpose of the caspaxos experiment, I dedicated the
    // first qp to loopback, next to primary consensus logic -- sharing a cq,
    // next and beyond to have their owq cq

    for (uint64_t n = 0; n < system_size_; ++n) {
      if (n == id_) continue;  // skip loopback
      for (uint64_t q = 1; q < num_qps_ + 1; ++q) {
        std::string key = MakeConnKey(id_, n, q);

        std::pair<std::unordered_map<std::string, ReliableConnection>::iterator,
                  bool>
            it;
        if (q <= 1) {
          it = connections_.emplace(key,
                                    ReliableConnection(dev, shared_cq_.get()));
        } else {
          it = connections_.emplace(key, ReliableConnection(dev));
        }
        it.first->second.Init(pd);
        registry_->Register<ConnInfo>(key,
                                      it.first->second.GetLocalPeerConnInfo());
        ROMULUS_DEBUG("QP with key={} has been registered.", key);
      }
    }

    // Publish the registered memory for this node
    for (const auto& i : *info_map) {
      std::string addr_key = MakeAddrKey(id_, memblock.GetBlockId(), i.first);
      if (!registry_->Register<AddrInfo>(addr_key, i.second)) {
        return false;
      }
      ROMULUS_DEBUG("Published AddrInfo {}", addr_key);
    }

    return true;
  }

  // Cross-node sense-reversing barrier using memcached
  bool WaitForNodesWithTimeout(std::chrono::seconds timeout) {
    uint64_t old;
    registry_->Fetch_and_Add(barrier_key_, 2, &old);
    uint64_t sense = old & 1;
    uint64_t expected_count = (system_size_ - 1) << 1 | sense;

    if (old == expected_count) {
      // We are the last to arrive, flip the barrier
      uint64_t new_val = 0 | (1 - sense);  // Reset counter and toggle sense
      registry_->Compare_and_Swap(barrier_key_, old + 2,
                                  new_val);  // optimistic
      return true;
    } else {
      auto start = std::chrono::steady_clock::now();
      while (std::chrono::steady_clock::now() - start < timeout) {
        Barrier cur;
        registry_->Retrieve<Barrier>(barrier_key_, &cur);
        if ((cur.counter & 1) != sense) {
          return true;  // Barrier passed
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      return false;  // Timeout
    }
  }

  // Cross-node sense-reversing barrier using memcached
  bool WaitForNodesNoTimeout() {
    uint64_t old;
    ROMULUS_ASSERT(registry_->Fetch_and_Add(barrier_key_, 2, &old),
                   "WaitForNodesNoTimeout() : FAA failed.");
    uint64_t sense = old & 1;
    uint64_t expected_count = (system_size_ - 1) << 1 | sense;
    // ROMULUS_DEBUG("Old={} Expected={}", old, expected_count);

    if (old == expected_count) {
      // We are the last to arrive, flip the barrier
      uint64_t new_val = 0 | (1 - sense);  // Reset counter and toggle sense
      ROMULUS_ASSERT(
          registry_->Compare_and_Swap(barrier_key_, old + 2, new_val),
          "WaitForNodesNoTimeout() : CAS failed.");
      return true;
    } else {
      while (1) {
        Barrier cur;
        registry_->Retrieve<Barrier>(barrier_key_, &cur);
        if ((cur.counter & 1) != sense) {
          break;  // Barrier passed
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      return true;
    }
  }

  bool WaitForNodesNoTimeout(int expected) {
    uint64_t old;
    ROMULUS_ASSERT(registry_->Fetch_and_Add(barrier_key_, 2, &old),
                   "WaitForNodesNoTimeout() : FAA failed.");
    uint64_t sense = old & 1;
    uint64_t expected_count = (expected - 1) << 1 | sense;
    // ROMULUS_DEBUG("Old={} Expected={}", old, expected_count);

    if (old == expected_count) {
      // We are the last to arrive, flip the barrier
      uint64_t new_val = 0 | (1 - sense);  // Reset counter and toggle sense
      ROMULUS_ASSERT(
          registry_->Compare_and_Swap(barrier_key_, old + 2, new_val),
          "WaitForNodesNoTimeout() : CAS failed.");
      return true;
    } else {
      while (1) {
        Barrier cur;
        registry_->Retrieve<Barrier>(barrier_key_, &cur);
        if ((cur.counter & 1) != sense) {
          break;  // Barrier passed
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      return true;
    }
  }

  void arrive_strict_barrier() {
    ROMULUS_ASSERT(WaitForNodesNoTimeout(),
                   "Failed while waiting for other nodes");
  }

  void arrive_strict_barrier(uint64_t num_nodes) {
    ROMULUS_ASSERT(WaitForNodesNoTimeout(num_nodes),
                   "Failed while waiting for other nodes");
  }

  void arrive_barrier_timeout() {
    ROMULUS_ASSERT(WaitForNodesWithTimeout(
                       std::chrono::seconds(ROMULUS_CONN_MGR_TIMEOUT_S)),
                   "Failed while waiting for other nodes");
  }

  bool Connect(const MemBlock& memblock) {
    // Connect loopback
    for(int i = 0; i < NUM_LOOPBACK_QPS; ++i) {
      uint64_t loopback_id;
      switch(i) {
        case 0:
          loopback_id = kLoopback_1;
          break;
        case 1:
          loopback_id = kLoopback_2;
          break;
        case 2:
          loopback_id = kLoopback_3;
          break;
        default:
          ROMULUS_FATAL("Invalid loopback index: {}", i);
          return false;
      }
      std::string host_key = MakeConnKey(id_, id_, loopback_id);
      auto it_loop = connections_.find(host_key);
      if (it_loop == connections_.end()) {
        ROMULUS_DEBUG("No connection matching key: {}", host_key);
        return false;
      }
      it_loop->second.Accept(it_loop->second.GetLocalPeerConnInfo());
      it_loop->second.Connect();
    }

    // Connect to all remote peers
    for (uint64_t n = 0; n < system_size_; ++n) {
      if (n == id_) continue;

      for (uint64_t q = 1; q < num_qps_ + 1; ++q) {
        std::string local_conn_key = MakeConnKey(id_, n, q);
        auto it_local = connections_.find(local_conn_key);
        if (it_local == connections_.end()) {
          ROMULUS_DEBUG("No connection matching key: {}", local_conn_key);
          return false;
        }

        std::string remote_conn_key = MakeConnKey(n, id_, q);
        ConnInfo conn_info;
        if (!registry_->Retrieve<ConnInfo>(remote_conn_key, &conn_info)) {
          ROMULUS_DEBUG("Failed to retrieve conn info: {}", remote_conn_key);
          return false;
        }
        it_local->second.Accept(conn_info);
        it_local->second.Connect();
      }
    }

    // Connect AddrInfo auto
    auto addr_infos = memblock.GetAllAddrInfos(false);
    for (uint64_t n = 0; n < system_size_; ++n) {
      if (n == id_) continue;
      for (const auto& region : *addr_infos) {
        std::string remote_addr_key =
            MakeAddrKey(n, memblock.GetBlockId(), region.first);
        if (!registry_->Retrieve<AddrInfo>(
                remote_addr_key, &(remote_addrs_[remote_addr_key].addr_info))) {
          ROMULUS_DEBUG("Failed to retrieve AddrInfo: {}", remote_addr_key);
          return false;
        }
      }
    }
    return true;
  }

  static std::string MakeConnKey(uint64_t local_id, uint64_t remote_id,
                                 uint64_t qp_id) {
    std::string key = "CONN@NODE";
    key += std::to_string(local_id);
    key += ":";
    key += std::to_string(remote_id);
    key += ":QP";
    key += std::to_string(qp_id);
    return key;
  }
  static std::string MakeAddrKey(uint64_t node_id, std::string_view memblock_id,
                                 std::string_view region_id) {
    std::string key = "ADDR@NODE";
    key += std::to_string(node_id);
    key += ":BLOCK";
    key += std::string(memblock_id);
    key += ":";
    key += std::string(region_id);
    return key;
  }

  static std::string MakeBarrierKey() { return "BARRIER"; }

  ReliableConnection* GetConnection(uint64_t remote_id, uint64_t qp_id) {
    std::string conn_key = MakeConnKey(id_, remote_id, qp_id);
    ROMULUS_DEBUG("Getting connection: {}", conn_key);
    auto iter = connections_.find(conn_key);
    if (iter != connections_.end()) return &(iter->second);
    return nullptr;
  }

  inline bool GetRemoteAddr(uint64_t remote_id, std::string block_id,
                            std::string region_id,
                            RemoteAddr* out_raddr) const {
    auto addr_key = MakeAddrKey(remote_id, block_id, region_id);
    ROMULUS_DEBUG("Getting remote address: {}", addr_key);
    auto iter = remote_addrs_.find(addr_key);
    if (iter != remote_addrs_.end()) {
      *out_raddr = iter->second;
      return true;
    }
    return false;
  }

 private:
  std::string host_;
  std::unordered_map<std::string, ReliableConnection> connections_;
  std::unordered_map<std::string, RemoteAddr> remote_addrs_;
  ConnectionRegistry* registry_;  //! NOT OWNED
  std::string barrier_key_;
  uint64_t id_;
  uint64_t system_size_;
  uint64_t num_qps_;
  CqUniquePtr shared_cq_;
};

}  // namespace romulus
