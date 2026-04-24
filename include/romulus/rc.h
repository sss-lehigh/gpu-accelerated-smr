#pragma once

#include <infiniband/verbs.h>

#include <cinttypes>
#include <memory>
#include <string>
#include <unordered_map>

#include "common.h"
#include "config.h"
#include "device.h"
#include "memblock.h"

#define POLL_CHUNK 8

namespace romulus {

class WorkRequest {
 public:
  WorkRequest() {
    std::memset(&sge_, 0, sizeof(ibv_sge));
    std::memset(&wr_, 0, sizeof(ibv_send_wr));
    next_ = nullptr;
  }

  WorkRequest* maybe_inline(uint32_t len, uint32_t max_inline) {
    if (len <= max_inline)
      wr_.send_flags |= IBV_SEND_INLINE;
    else
      wr_.send_flags &= ~IBV_SEND_INLINE;
    return this;
  }

  static void BuildWrite(const LocalAddr& local, const RemoteAddr& remote,
                         uint64_t wr_id, WorkRequest* wr) {
    BuildSignaled(local, remote, wr_id, wr);
    wr->opcode(IBV_WR_RDMA_WRITE);
    wr->opcode(IBV_WR_RDMA_WRITE)
        ->maybe_inline(local.length, ROMULUS_RC_MAX_INLINE);
  }

  static void BuildRead(const LocalAddr& local, const RemoteAddr& remote,
                        uint64_t wr_id, WorkRequest* wr) {
    BuildSignaled(local, remote, wr_id, wr);
    wr->opcode(IBV_WR_RDMA_READ);
  }

  static void BuildCAS(const LocalAddr& local, const RemoteAddr& remote,
                       uint64_t compare, uint64_t swap, uint64_t wr_id,
                       WorkRequest* wr) {
    BuildSignaled(local, remote, wr_id, wr);
    wr->cas(compare, swap);
  }

  static void BuildSignaled(const LocalAddr& local, const RemoteAddr& remote,
                            uint64_t wr_id, WorkRequest* wr) {
    wr->local_addr(local)->remote_addr(remote)->signaled()->wr_id(wr_id);
  }

  WorkRequest* local_addr(const LocalAddr& local) {
    sge_.addr = local.addr + local.offset;
    sge_.length = local.length;
    sge_.lkey = local.key;
    wr_.num_sge = 1;
    wr_.sg_list = &sge_;
    return this;
  }

  WorkRequest* remote_addr(const RemoteAddr& remote) {
    wr_.wr.rdma.remote_addr = remote.addr_info.addr + remote.addr_info.offset;
    wr_.wr.rdma.rkey = remote.addr_info.key;
    return this;
  }

  // A bit wonky but this should be done after setting the local and remote
  // addresses.
  WorkRequest* cas(uint64_t compare, uint64_t swap) {
    wr_.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
    wr_.wr.atomic.remote_addr = wr_.wr.rdma.remote_addr;
    wr_.wr.atomic.rkey = wr_.wr.rdma.rkey;
    wr_.wr.atomic.compare_add = compare;
    wr_.wr.atomic.swap = swap;
    return this;
  }

  WorkRequest* opcode(ibv_wr_opcode code) {
    wr_.opcode = code;
    return this;
  }

  WorkRequest* signaled() {
    wr_.send_flags |= IBV_SEND_SIGNALED;
    return this;
  }

  WorkRequest* unsignaled() {
    wr_.send_flags &= ~(IBV_SEND_SIGNALED);
    return this;
  }

  WorkRequest* wr_id(uint64_t id) {
    wr_.wr_id = id;
    return this;
  }

  WorkRequest* append(WorkRequest* wr) {
    next_ = wr;
    this->wr_.next = wr->wr_ptr();
    return this;
  }

  ibv_send_wr* wr_ptr() { return &wr_; }
  WorkRequest* next() { return next_; }

 private:
  ibv_sge sge_;
  ibv_send_wr wr_;
  WorkRequest* next_;
};

enum PERM_FLAGS {
  LOCAL_READ = 0,
  LOCAL_WRITE = IBV_ACCESS_LOCAL_WRITE,
  REMOTE_READ = IBV_ACCESS_REMOTE_READ,
  REMOTE_WRITE = IBV_ACCESS_REMOTE_WRITE,
  REMOTE_ATOMIC = IBV_ACCESS_REMOTE_ATOMIC,
  RELAXED_ORDERING = IBV_ACCESS_RELAXED_ORDERING,
  ON_DEMAND_PAGING = IBV_ACCESS_ON_DEMAND
};

//+ Is it worth having different connection "types" to encode different
//+ underlying memory structures?
class ReliableConnection {
 public:
  // Default constructor necessary for use with boost::icl::interval_map
  ReliableConnection() {
    port_ = -1;
    lid_ = 0;
    outstanding_ = 0;
    cq_raw_ = nullptr;
  }
  explicit ReliableConnection(const Device& dev, ibv_cq* shared_cq = nullptr) {
    outstanding_ = 0;
    port_ = dev.GetPort();
    lid_ = dev.GetLid();

    kMaxWr_ = ROMULUS_RC_MAX_WR;
    kMaxSge_ = ROMULUS_RC_MAX_SGE;
    kMaxInline_ = ROMULUS_RC_MAX_INLINE;
    auto* context = dev.GetContext();
    ROMULUS_ASSERT(ibv_query_gid(context, port_, 0, &gid_) == 0,
                   "Failed to query GID from device");

    // Populate init attributes
    memset(&init_attr_, 0, sizeof(init_attr_));
    init_attr_.qp_type = IBV_QPT_RC;
    init_attr_.cap.max_send_wr = kMaxWr_;
    init_attr_.cap.max_recv_wr = kMaxWr_;
    init_attr_.cap.max_send_sge = kMaxSge_;
    init_attr_.cap.max_recv_sge = kMaxSge_;
    init_attr_.cap.max_inline_data = kMaxInline_;
    ROMULUS_DEBUG("Creating RC with max_wr={}, max_sge={}, max_inline={}",
                  kMaxWr_, kMaxSge_, kMaxInline_);

    // Create a cq for this connection
    if (shared_cq) {
      cq_raw_ = shared_cq;
    } else {
      cq_.reset(
          ibv_create_cq(context, ROMULUS_RC_CQ_SIZE, nullptr, nullptr, 0));
      cq_raw_ = cq_.get();
    }
    init_attr_.send_cq = cq_raw_;
    init_attr_.recv_cq = cq_raw_;
  }

  // Creates the underlying QP and modifies it to be in the INIT state.
  bool Init(ibv_pd* pd) {
    ROMULUS_DEBUG("[{}] Initializaing reliable connection",
                  reinterpret_cast<uintptr_t>(this));
    if (init_attr_.send_cq == nullptr || init_attr_.recv_cq == nullptr) {
      ROMULUS_FATAL("Cannot initialize a connection without CQs");
      abort();
    }
    qp_.reset(ibv_create_qp(pd, &init_attr_));
    if (qp_ == nullptr) {
      ROMULUS_FATAL("Failed to create QP for connection: {}",
                    reinterpret_cast<uintptr_t>(this));
      return false;
    }

    // CAN REMOVE -- verbose logging to check that the QP was created with the
    // requested capabilities
    // {
    //   struct ibv_qp_attr attr;
    //   struct ibv_qp_init_attr qinit;
    //   memset(&attr, 0, sizeof(attr));
    //   memset(&qinit, 0, sizeof(qinit));
    //   int rc = ibv_query_qp(qp_.get(), &attr, IBV_QP_CAP, &qinit);
    //   if (rc == 0) {
    //     ROMULUS_INFO("QP caps: requested_inline={}, actual_inline={},
    //     max_send_wr={}, max_send_sge={}",
    //                  kMaxInline_, qinit.cap.max_inline_data,
    //                  qinit.cap.max_send_wr, qinit.cap.max_send_sge);
    //   } else {
    //     ROMULUS_INFO("ibv_query_qp failed: {}", std::strerror(errno));
    //   }
    // }

    local_peer_.qp_num = qp_->qp_num;
    local_peer_.lid = lid_;
    *(reinterpret_cast<uint64_t*>(&local_peer_.gid[0])) =
        *(reinterpret_cast<const uint64_t*>(&gid_.raw[0]));
    *(reinterpret_cast<uint64_t*>(&local_peer_.gid[8])) =
        *(reinterpret_cast<const uint64_t*>(&gid_.raw[8]));

    // Transition to initialized state
    memset(&conn_attr_, 0, sizeof(conn_attr_));
    conn_attr_.qp_state = IBV_QPS_INIT;
    conn_attr_.port_num = port_;
    conn_attr_.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    unsigned int mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    ROMULUS_ASSERT(ibv_modify_qp(qp_.get(), &conn_attr_, mask) == 0,
                   "Failed to modify QP to INIT");
    return true;
  }

  // Modifies the QP to accept messages from `remote_conn`. Also, takes
  // ownership of `remote_conn`.
  bool Accept(const ConnInfo& remote_conn) {
    ROMULUS_DEBUG("[{}] Accepting remote connection",
                  reinterpret_cast<uintptr_t>(this));
    // Transition to RTR
    //+ Check over configuration
    memset(&conn_attr_, 0, sizeof(conn_attr_));
    conn_attr_.qp_state = IBV_QPS_RTR;
    conn_attr_.path_mtu = IBV_MTU_1024;
    conn_attr_.dest_qp_num = remote_conn.qp_num;
    conn_attr_.rq_psn = ROMULUS_RC_DEFAULT_PSN;
    conn_attr_.max_dest_rd_atomic = ROMULUS_RC_MAX_RD_ATOMIC;
    conn_attr_.min_rnr_timer = 12;

    conn_attr_.ah_attr.dlid = remote_conn.lid;
    conn_attr_.ah_attr.port_num = port_;

    conn_attr_.ah_attr.is_global = 1;
    std::memcpy(conn_attr_.ah_attr.grh.dgid.raw, remote_conn.gid, 16);
    conn_attr_.ah_attr.grh.sgid_index = 0;
    conn_attr_.ah_attr.grh.hop_limit = 4;

    unsigned int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    ROMULUS_ASSERT(ibv_modify_qp(qp_.get(), &conn_attr_, mask) == 0,
                   "Failed to modify QP to RTR: {}", errno);
    remote_peer_ = remote_conn;
    return true;
  }

  // Modifies the QP to send messages to the remote connection it receives
  // them from (i.e., `remote_conn` above).
  bool Connect() {
    ROMULUS_DEBUG("[{}] Connecting", reinterpret_cast<uintptr_t>(this));
    // Transition to RTS
    memset(&conn_attr_, 0, sizeof(conn_attr_));
    conn_attr_.qp_state = IBV_QPS_RTS;
    conn_attr_.timeout = 12;
    conn_attr_.retry_cnt = 6;
    conn_attr_.rnr_retry = 0;
    conn_attr_.sq_psn = ROMULUS_RC_DEFAULT_PSN;
    conn_attr_.max_rd_atomic = ROMULUS_RC_MAX_RD_ATOMIC;

    int mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
               IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

    ROMULUS_ASSERT(ibv_modify_qp(qp_.get(), &conn_attr_, mask) == 0,
                   "Failed to modify QP to RTS");
    return true;
  }

  bool Post(WorkRequest* head, uint32_t num_wrs) {
    ibv_send_wr* bad_request = nullptr;
    if (ibv_post_send(qp_.get(), head->wr_ptr(), &bad_request) != 0) {
      ROMULUS_FATAL("Failed to post: error={}", std::strerror(errno));
      return false;
    }
    outstanding_ += num_wrs;
    return true;
  }

  bool Read(const LocalAddr& local, const RemoteAddr& remote, uint64_t wr_id) {
    ibv_send_wr* bad_request = nullptr;
    WorkRequest request;
    WorkRequest::BuildRead(local, remote, wr_id, &request);

    // ROMULUS_ASSERT(
    //     ibv_post_send(qp_.get(), request.wr_ptr(), &bad_request) == 0,
    //     "Failed to post READ op");

    if (ibv_post_send(qp_.get(), request.wr_ptr(), &bad_request) != 0) {
      ROMULUS_DEBUG("Failed to post READ op: error={}", std::strerror(errno));
      return false;
    }
    // ROMULUS_DEBUG("READ: (qp={}, addr={:#x}, len={}, wr_id={})",
    //               remote.conn_info.qp_num, remote.addr_info.addr,
    //               remote.addr_info.length, wr_id);
    ++outstanding_;
    return true;
  }

  bool Write(const LocalAddr& local, const RemoteAddr& remote, uint64_t wr_id) {
    ibv_send_wr* bad_request = nullptr;
    WorkRequest request;
    WorkRequest::BuildWrite(local, remote, wr_id, &request);

    ROMULUS_ASSERT(
        ibv_post_send(qp_.get(), request.wr_ptr(), &bad_request) == 0,
        "Failed to post WRITE (wr_id={}) err: {}", wr_id, std::strerror(errno));
    // ROMULUS_DEBUG(
    //     "WRITE: (qp={}, addr={:#x}, len={}, wr_id={})",
    //     remote.conn_info.qp_num, remote.addr_info.addr +
    //     remote.addr_info.offset, local.length, wr_id);
    ++outstanding_;
    return true;
  }

  bool CompareAndSwap(const LocalAddr& local, const RemoteAddr& remote,
                      uint64_t compare, uint64_t swap, uint64_t wr_id) {
    ibv_send_wr* bad_request = nullptr;
    WorkRequest request;
    WorkRequest::BuildCAS(local, remote, compare, swap, wr_id, &request);

    ROMULUS_ASSERT(
        ibv_post_send(qp_.get(), request.wr_ptr(), &bad_request) == 0,
        "Failed to post CAS op");
    ROMULUS_DEBUG("CAS: (qp={}, addr={:#x}, compare={}, swap={}, wr_id={})",
                  remote.conn_info.qp_num, remote.addr_info.addr, compare, swap,
                  wr_id);
    ++outstanding_;
    return true;
  }

  void Reset() {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RESET;
    auto ret = ibv_modify_qp(qp_.get(), &attr, IBV_QP_STATE);
    if (ret) {
      ROMULUS_FATAL("Could not modify QP to RESET: ",
                    std::string(std::strerror(errno)));
    }
  }

  template <typename... PERM_FLAGS>
  void ChangePermissions(ibv_pd* pd, PERM_FLAGS... flags) {
    int combined_flag = (static_cast<uint32_t>(flags) | ...);
    auto apply_perm = [&combined_flag, this]() {
      struct ibv_qp_attr attr;
      memset(&attr, 0, sizeof(attr));

      attr.qp_access_flags = static_cast<unsigned>(combined_flag);
      // verb that actually modifies the access rights to the qp
      auto ret = ibv_modify_qp(qp_.get(), &attr, IBV_QP_ACCESS_FLAGS);
      return ret == 0;
    };
    // If for whatever reason our attempt to change acces rights in the QP
    // fails, then we fallback to reseting to QP with default flags
    if (!apply_perm()) {
      ROMULUS_DEBUG("Normal access change failed, invoking fallback...");
      Reset();
      Init(pd);
      Connect();
    }
  }

  int PollBatch(const std::vector<uint64_t>& wr_ids) {
    uint32_t posted = wr_ids.size();
    uint32_t completions = 0;
    wcs_.resize(posted);
    auto timeout = std::chrono::milliseconds(100);
    auto start_time = std::chrono::steady_clock::now();

    while (completions < posted &&
           std::chrono::steady_clock::now() - start_time < timeout) {
      // ROMULUS_DEBUG("Polling for completions: posted={}, completions={}",
      // posted,
      //               completions);
      ibv_wc tmp[POLL_CHUNK];
      int n = ibv_poll_cq(cq_raw_, POLL_CHUNK, tmp);
      if (n < 0) {
        ROMULUS_FATAL("Failed to poll shared CQ: {}", std::strerror(errno));
        return completions;
      }
      for (int j = 0; j < n; ++j) {
        for (uint32_t k = 0; k < posted; ++k) {
          if (tmp[j].wr_id == wr_ids[k]) {
            wcs_[completions++] = tmp[j];
            break;
          }
        }
      }
    }
    return completions;
  }

  // Poll for a single completion, thread-safe
  int PollOnce(std::atomic<int>* ack) {
    // poll for a single completion
    ibv_wc wc;
    while (*ack != 0) {
      int poll = ibv_poll_cq(cq_raw_, 1, &wc);
      if (poll == 0 || (poll < 0 && errno == EAGAIN)) continue;
      if (wc.status != IBV_WC_SUCCESS) {
        ROMULUS_ASSERT(
            wc.status == IBV_WC_SUCCESS, "ibv_poll_cq(): {} ({})",
            (poll < 0 ? strerror(errno) : ibv_wc_status_str(wc.status)),
            (std::stringstream() << wc.wr_id).str());
      }
      int old = ((std::atomic<int>*)wc.wr_id)->fetch_sub(1);
      ROMULUS_ASSERT(old >= 1, "Broken synchronization");
    }
    return 0;
  }

  // Try once to poll for the outstanding completions.
  int TryProcessOutstanding() {
    if (outstanding_ == 0) return 0;
    // ROMULUS_ASSERT([]() { return -1; }, outstanding_ > 0, "No outstanding
    // requests."); ROMULUS_ASSERT(ROMULUS_ABORT, outstanding_ > 0, "No
    // outstanding requests.");

    // Poll for at most the expepected number of comps less the running total
    wcs_.resize(outstanding_);
    num_wcs_ = ibv_poll_cq(cq_raw_, outstanding_, &wcs_[0]);
    if (num_wcs_ == -1) {
      ROMULUS_FATAL("Failed to poll for compeltions: error={}",
                    std::strerror(errno));
      return -1;
    } else {
      for (int i = 0; i < num_wcs_; ++i) {
        auto wc = wcs_[i];
        // ROMULUS_DEBUG("Got completion: qp={}, type={}, wr_id={}, status={}",
        //               wc.qp_num, (int)wc.opcode, wc.wr_id,
        //               ibv_wc_status_str(wc.status));
        if (wc.status != IBV_WC_SUCCESS) {
          ROMULUS_FATAL("Work request (wc={}) failed: {}", wc.wr_id,
                        ibv_wc_status_str(wc.status));
        }
      }
      outstanding_ -= num_wcs_;
      return num_wcs_;
    }
  }

  bool CheckCompletionsForId(uint64_t wr_id) {
    for (int i = 0; i < num_wcs_; ++i) {
      if (wcs_[i].wr_id == wr_id) return true;
    }
    return false;
  }

  int ProcessOutstanding() {
    ROMULUS_ASSERT(outstanding_ > 0, "No outstanding requests.");
    uint32_t total_comps = 0;
    int num_comps;
    wcs_.resize(outstanding_);
    while (outstanding_ > 0) {
      num_comps = TryProcessOutstanding();
      total_comps += num_comps;
    }
    return total_comps;
  }

  // Only returns the number of completions matching wr_id.
  int ProcessOutstanding(uint64_t wr_id) {
    ROMULUS_ASSERT(outstanding_ > 0, "No outstanding requests.");
    uint32_t total_comps = 0;
    int num_comps;
    wcs_.resize(outstanding_);
    while (total_comps < outstanding_) {
      // Poll for at most the expepected number of comps less the running total
      num_comps =
          ibv_poll_cq(cq_raw_, outstanding_ - total_comps, &wcs_[total_comps]);
      if (num_comps == -1) {
        ROMULUS_FATAL("Failed to poll for completions");
        return total_comps;
      }
      total_comps += num_comps;
    }

    int adjusted_comps = 0;
    for (uint32_t i = 0; i < total_comps; ++i) {
      if (wcs_[i].wr_id != wr_id) continue;
      if (wcs_[i].status != IBV_WC_SUCCESS) {
        ROMULUS_FATAL("Work request (wc={}) failed: {}", wcs_[i].wr_id,
                      ibv_wc_status_str(wcs_[i].status));
      }
      ++adjusted_comps;
    }
    outstanding_ -= total_comps;
    return adjusted_comps;
  }

  int ProcessCompletions(uint32_t expected_comps) {
    int num_comps;
    uint32_t total_comps = 0;
    wcs_.resize(expected_comps);
    auto start_time = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(100);

    while (total_comps < expected_comps &&
           std::chrono::steady_clock::now() - start_time < timeout) {
      // Poll for at most the expepected number of comps less the running total
      num_comps = ibv_poll_cq(cq_raw_, expected_comps - total_comps,
                              &wcs_[total_comps]);
      if (num_comps == -1) {
        ROMULUS_FATAL("Failed to poll for compeltions");
        break;
      }
      total_comps += num_comps;
    }

    for (uint32_t i = 0; i < total_comps; ++i) {
      if (wcs_[i].status != IBV_WC_SUCCESS) {
        // ROMULUS_FATAL("Work request (wc={}) failed: {}", wcs_[i].wr_id,
        //               ibv_wc_status_str(wcs_[i].status));
        ROMULUS_DEBUG("Work request (wc={}) failed: {}", wcs_[i].wr_id,
                      ibv_wc_status_str(wcs_[i].status));
      }
    }
    outstanding_ -= total_comps;
    return total_comps;
  }

  ConnInfo GetLocalPeerConnInfo() const { return local_peer_; }

  ConnInfo GetRemotePeerConnInfo() const { return remote_peer_; }

  // Return the GID of this conection
  union ibv_gid GetGid() const { return gid_; }

  // Return the QP number of this connection
  uint32_t GetQpNum() const { return qp_->qp_num; }

  // Return the LID of this connection
  uint32_t GetLid() const { return lid_; }

  ibv_cq* GetCQ() const { return cq_raw_; }

  ibv_qp* GetQP() const { return qp_.get(); }

  bool operator==(const ReliableConnection& rhs) const {
    return (qp_->qp_num == rhs.qp_->qp_num &&
            *reinterpret_cast<uint64_t*>(gid_.raw[0]) ==
                *reinterpret_cast<uint64_t*>(rhs.gid_.raw[0]) &&
            *reinterpret_cast<uint64_t*>(gid_.raw[8]) ==
                *reinterpret_cast<uint64_t*>(rhs.gid_.raw[8]) &&
            lid_ == rhs.lid_);
  }

 private:
  //! The declaration of `cq_` must come before `qp_`, otherwise the
  //! completion queue will be destroyed before the queue pair and cause a
  //! memory leak.

  // A unique pointer to the completion queue associated with the send and
  // receive queues.
  CqUniquePtr cq_;

  ibv_cq* cq_raw_;

  // A unique pointer to the Queue Pair that represents this connection.
  QpUniquePtr qp_;

  // Struct passed during initialization that defines important startup values
  // (e.g., pointers to the completion queues, maximum number of outstanding
  // work requests, etc.)
  struct ibv_qp_init_attr init_attr_;

  // Struct passed during QP manipulation to define the state transition.
  struct ibv_qp_attr conn_attr_;

  // Global identifier of the context associated with `memblock_`. This is
  // necessary when registering the connection.
  union ibv_gid gid_;

  // Port of the RNIC associated with this connection.
  int port_;

  // LID assigned by the subnet manager
  uint32_t lid_;

  // Local connection information
  ConnInfo local_peer_;

  // The current remote peer accepted at this connection.
  ConnInfo remote_peer_;

  // Number of outstanding work requests.
  uint32_t outstanding_;

  // Vector to hold completions when polling.
  std::vector<ibv_wc> wcs_;
  int num_wcs_;

  // Connection limits
  uint32_t kMaxWr_;
  uint32_t kMaxSge_;
  uint32_t kMaxInline_;
};

}  // namespace romulus
