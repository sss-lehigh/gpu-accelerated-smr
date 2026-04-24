//* QP configs
#define ROMULUS_RC_MAX_WR 8192 // 64        // Maximum oustanding WRs
#define ROMULUS_RC_MAX_SGE 16        // Maximum scatter-gather elements
#define ROMULUS_RC_MAX_INLINE 448    // Maximum size of inlined data
#define ROMULUS_RC_DEFAULT_PSN 1000  // Default packet sequence number
#define ROMULUS_RC_MAX_RD_ATOMIC 16  // Maximum outstanding atomic read ops

//* CQ configs
#define ROMULUS_RC_CQ_SIZE 16384 // 256  // Minimum number of CQE in CQ

//* MR configs
#define ROMULUS_MEMBLOCK_MR_SIZE 1024  // Default memory region size

//* Connection manager configs
#define ROMULUS_SLEEP_US 10

//* Directory configs
#define ROMULUS_MAX_DIR_SIZE_MB 10  // Max directory footprint in MB

//* Connection manager configs
#define ROMULUS_CONN_MGR_TIMEOUT_S 10