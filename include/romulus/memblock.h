#pragma once

#include <infiniband/verbs.h>

#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include "common.h"

namespace romulus {

// A MemBlock is a block of memory registered with the RNIC for remote access.
// It is associated with a single protection domain (and therefore QP) but can
// be broken into several (possibly overlapping) memory regions that are
// remotely accessible.
class MemBlock {
 public:
  MemBlock() : block_id_(""), raw_(nullptr), size_(0), pd_(nullptr) {}
  MemBlock(std::string block_id, struct ibv_pd* pd, uint8_t* buf, uint64_t size)
      : block_id_(std::move(block_id)), raw_(buf), size_(size), pd_(pd) {
    ROMULUS_INFO("Instantiating Memblock with {} bytes", size);
    memset(raw_, 0, size_);
  }

  // Register a new memory region with this MemBlock with `region_id` starting
  // at `offset` from the base address of the raw memory backing the MemBlock
  // and of size `len`.
  bool RegisterMemRegion(std::string region_id, size_t offset, size_t len) {
    ROMULUS_ASSERT(offset + len <= size_,
                   "Attempted to register memory out of bounds: "
                   "memblock={:x}, offset={}, len={}, size={}",
                   reinterpret_cast<uintptr_t>(this), offset, len, size_);

    if (mrs_.find(region_id) != mrs_.end()) {
      return false;  // already registered
    }

    unsigned int access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    mrs_[region_id] =
        MrUniquePtr(ibv_reg_mr(pd_, raw_ + offset, len, access_flags));
    if (mrs_[region_id] == nullptr) {
      ROMULUS_FATAL("Failed to register memory region: {}, error={}", region_id,
                    std::strerror(errno));
      return false;
    }

    if(mrs_[region_id]->addr != static_cast<void*>(raw_ + offset)){
      ROMULUS_DEBUG("Address mismatch in MR registration: expected={}, actual={:x}",
        reinterpret_cast<uintptr_t>(raw_ + offset), reinterpret_cast<uintptr_t>(mrs_[region_id]->addr));
    }
    return true;
  }

  // Return the memory region associated with `region_id`. If none exists,
  // return `nullptr`.
  struct ibv_mr* GetMemRegion(std::string region_id) {
    if (mrs_.find(region_id) != mrs_.end()) {
      return mrs_[region_id].get();
    }
    return nullptr;
  }

  int GetSize() const { return size_; }
  ibv_pd* GetPd() const { return pd_; }
  uint8_t* GetRaw() const { return raw_; }
  std::string GetBlockId() const { return block_id_; }

  // Create an AddrInfo for a region with optional offset/length
  bool MakeAddrInfoForRemoteOp(std::string region_id,
                               std::optional<uint32_t> offset,
                               std::optional<uint32_t> length,
                               AddrInfo* info) const {
    auto* mr = mrs_.at(region_id).get();
    ROMULUS_ASSERT(offset.value_or(0) + length.value_or(0) <= mr->length || offset.value_or(0) + length.value_or(0) <= mr->length,
                   "Invalid offset/length: {} + {} > {}", offset.value_or(0),
                   length.value_or(0), mr->length);
    

    info->addr = reinterpret_cast<uint64_t>(
        reinterpret_cast<uint8_t*>(mr->addr) + offset.value_or(0));
    info->offset = offset.value_or(0);
    info->length = length.value_or(mr->length - offset.value_or(0));
    info->key = mr->rkey;
    return true;
  }

  // Return an AddrInfo for the whole region
  AddrInfo GetAddrInfo(std::string region_id) const {
    AddrInfo info;
    auto* mr = mrs_.at(region_id).get();
    info.addr = reinterpret_cast<uint64_t>(mr->addr);
    info.offset = 0;
    info.length = mr->length;
    info.key = mr->lkey;
    return info;
  }

  // Return AddrInfos for all registered regions
  std::unique_ptr<std::unordered_map<std::string, AddrInfo>> GetAllAddrInfos(
      bool for_remote) const {
    auto info_map =
        std::make_unique<std::unordered_map<std::string, AddrInfo>>();
    for (const auto& mr : mrs_) {
      (*info_map).emplace(
          mr.first, AddrInfo{reinterpret_cast<uint64_t>(mr.second->addr), 0,
                             static_cast<uint32_t>(mr.second->length),
                             for_remote ? mr.second->rkey : mr.second->lkey});
    }
    return info_map;
  }

  //! Test helpers
  bool WriteRawForTest(const uint8_t* bytes, size_t offset, size_t len) {
    ROMULUS_ASSERT(offset + len <= size_, "WriteRawForTest OOB");
    memcpy(raw_ + offset, bytes, len);
    return true;
  }

  bool ReadRawForTest(size_t offset, size_t len, uint8_t* bytes) {
    ROMULUS_ASSERT(offset + len <= size_, "ReadRawForTest OOB");
    memcpy(bytes, raw_ + offset, len);
    return true;
  }

 private:
  std::string block_id_;  // Unique ID per node
  uint8_t* raw_;          // Backing memory
  size_t size_;           // Block size
  struct ibv_pd* pd_;     // Not owned

  std::unordered_map<std::string, MrUniquePtr> mrs_;  // region_id -> MR
};

}  // namespace romulus
