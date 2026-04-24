#pragma once

#include <boost/icl/interval_map.hpp>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "common/common.h"
#include "connection/rc.h"
#include "connection_manager/connection_manager.h"
#include "memblock/memblock.h"

namespace romulus {

/* =======================
 *  Entry State + Entry
 * ======================= */

// Possible states of a directory entry
enum EntryState : uint8_t { Undefined, Unshared, Shared, Dirty, Locked };

class DirectoryEntry {
 public:
  DirectoryEntry() : host_(0), offset_(0) {
    state_.entry_state = EntryState::Undefined;
    state_.bloom_filter = 0;
  }

  DirectoryEntry(uint32_t host, uint32_t offset, EntryState state)
      : host_(host), offset_(offset) {
    static_assert(sizeof(state_) == sizeof(uint64_t), "Unexpected state size");
    state_.entry_state = state;
    state_.bloom_filter = 0;
  }

  static DirectoryEntry UndefinedEntry() {
    return DirectoryEntry(0, 0, EntryState::Undefined);
  }

 private:
  uint32_t host_;    // NodeId of the owner
  uint32_t offset_;  // Offset in the canonical store

  // Remotely CAS'able state word (64 bits total)
  union {
    uint64_t as_uint64;
    struct {
      EntryState entry_state;
      uint64_t bloom_filter : 48;
    } __attribute__((packed));
  } state_;
};

/* =======================
 *  Directory Keys
 * ======================= */

namespace {
inline uint64_t KeyToIndex(uint32_t key) { return key; }
inline uint64_t KeyToIndex(uint64_t key) { return key; }
inline uint64_t KeyToIndex(const std::string& key) {
  return std::hash<std::string>{}(key);
}
}  // namespace

template <typename T>
class DirectoryKey {
 public:
  bool operator<(const DirectoryKey<T>& rhs) const {
    return this->key_ < rhs.key_;
  }
  DirectoryKey& operator++() {
    this->key_++;
    return *this;
  }
  DirectoryKey& operator--() {
    this->key_--;
    return *this;
  }
  uint64_t hash() const { return KeyToIndex(key_); }

 private:
  T key_;
};

/* =======================
 *  Directory
 * ======================= */

// A distributed directory that maps keys to owning nodes.
// Stores the local slice and remote intervals in an interval_map.
template <typename KeyType>
class Directory {
  using DirectoryKeyType = DirectoryKey<KeyType>;

 public:
  explicit Directory(ConnectionManager* mgr, const MemBlock* memblock,
                     std::string_view region_id);

  void AddRemoteDirectory(
      boost::icl::discrete_interval<DirectoryKeyType> interval,
      std::string_view node_id, std::string_view block_id);

  DirectoryEntry Lookup(DirectoryKeyType key);

  bool TryAcquire(DirectoryKeyType key, EntryState* prev);
  void Release(DirectoryKeyType key, const EntryState state);

 private:
  int GetKeyIndex(DirectoryKeyType);
  DirectoryEntry RemoteLookup(DirectoryKeyType key);

  //! NOT OWNED
  const MemBlock* memblock_;
  const std::string region_id_;
  const int size_;
  ConnectionManager* mgr_;
  Device* dev_;

  boost::icl::interval_map<DirectoryKeyType, ReliableConnection*>
      connection_imap_;
};

/* =======================
 *  Inline Implementations
 * ======================= */

template <typename KeyType>
Directory<KeyType>::Directory(ConnectionManager* mgr, const MemBlock* memblock,
                              std::string_view region_id)
    : memblock_(memblock),
      region_id_(region_id),
      mgr_(mgr),
      size_(memblock_->GetAddrInfo(region_id_).length) {}

template <typename KeyType>
void Directory<KeyType>::AddRemoteDirectory(
    boost::icl::discrete_interval<DirectoryKeyType> interval,
    std::string_view node_id, std::string_view block_id) {
  auto conn = mgr_->GetConnection(node_id, block_id);
  connection_imap_.insert(std::make_pair(interval, conn));
}

template <typename KeyType>
DirectoryEntry Directory<KeyType>::Lookup(DirectoryKeyType key) {
  auto conn = connection_imap_.find(key);
  // TODO: implement actual lookup logic (local vs remote)
  return DirectoryEntry::UndefinedEntry();
}

template <typename KeyType>
bool Directory<KeyType>::TryAcquire(DirectoryKeyType key, EntryState* prev) {
  // TODO: implement locking
  return false;
}

template <typename KeyType>
void Directory<KeyType>::Release(DirectoryKeyType key, const EntryState state) {
  // TODO: implement unlock/release
}

template <typename KeyType>
int Directory<KeyType>::GetKeyIndex(DirectoryKeyType) {
  // TODO: implement indexing logic
  return 0;
}

template <typename KeyType>
DirectoryEntry Directory<KeyType>::RemoteLookup(DirectoryKeyType key) {
  // TODO: implement remote lookup
  return DirectoryEntry::UndefinedEntry();
}

}  // namespace romulus
