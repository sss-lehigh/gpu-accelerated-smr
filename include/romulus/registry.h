#pragma once

#include <libmemcached/memcached.h>

#include <chrono>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <typeinfo>

#include "common.h"
#include "rc.h"

namespace romulus {

class ConnectionRegistry {
 public:
  ConnectionRegistry(std::string_view name, std::string_view registry_ip)
      : name_(name),
        barrier_key_(MakeInternalKey("barrier")),
        wait_failed_(false) {
    auto config_str = std::string("--BINARY-PROTOCOL --SERVER=") +
                      std::string(registry_ip) + ":9999";
    memc_client_.reset(memcached(config_str.data(), config_str.size()));
  }

  template <typename T>
  bool Register(std::string_view key, const T& info) {
    std::string value = info.ToString();
    std::string key_internal = MakeInternalKey(key);
    memcached_return_t rc =
        memcached_set(memc_client_.get(), key_internal.data(),
                      key_internal.size(), value.data(), value.size(), 0, 0);

    if (rc != MEMCACHED_SUCCESS) {
      ROMULUS_FATAL("Failed to register key: {} <{}> = {}", key_internal,
                    typeid(info).name(), value);
      return false;
    }
    ROMULUS_DEBUG("Registered key: {} <{}> = {}", key_internal,
                  typeid(info).name(), value);
    return true;
  }

  template <typename T>
  bool Retrieve(std::string_view key, T* info) {
    size_t length;
    memcached_return_t rc;
    std::string key_internal = MakeInternalKey(key);
    char* buf = memcached_get(memc_client_.get(), key_internal.data(),
                              key_internal.size(), &length, 0, &rc);
    if (rc != MEMCACHED_SUCCESS) {
      ROMULUS_FATAL("Remote retrieval of key failed: {} <{}>", key_internal,
                    typeid(info).name());
      return false;
    }
    info->FromString(std::string(buf));
    // ROMULUS_DEBUG("Retrieved key: {} <{}> = {}", key_internal,
    //               typeid(info).name(), info->ToString());
    free(buf);
    return true;
  }

  template <typename T>
  bool Set(std::string_view key, const T& value) {
    std::string key_str = MakeInternalKey(key);
    std::string value_str = value.ToString();

    memcached_return_t rc =
        memcached_set(memc_client_.get(), key_str.data(), key_str.size(),
                      value_str.data(), value_str.size(), 0, 0);
    if (rc != MEMCACHED_SUCCESS) {
      ROMULUS_FATAL("Failed to set key: {} = {}", key_str, value_str);
      return false;
    }
    ROMULUS_DEBUG("Set key: {} = {}", key_str, value_str);
    return true;
  }

  bool Fetch_and_Add(std::string_view key, uint64_t delta, uint64_t* old_value = nullptr) {
    std::string key_str = MakeInternalKey(key);
    uint64_t new_val = 0;
    memcached_return_t rc =
        memcached_increment_with_initial(memc_client_.get(), key_str.data(),
                                         key_str.size(), delta, 0, 0, &new_val);
    if (rc != MEMCACHED_SUCCESS) {
      const char* err = memcached_strerror(memc_client_.get(), rc);
        ROMULUS_FATAL("Failed to FAA key: {} by {} (rc={}, msg={})", key_str, delta, static_cast<int>(rc), err);
        return false;
    }
    if (old_value != nullptr)
      *old_value = new_val - delta;
    // ROMULUS_DEBUG("FAA key: {} by {} (new={})", key_str, delta, new_val);
    return true;
  }

  bool Compare_and_Swap(std::string_view key, uint64_t expected, uint64_t new_value) {
    std::string key_str = MakeInternalKey(key);

    memcached_return_t rc;
    size_t val_len = 0;
    uint32_t cas_token = 0;

    char* buf = memcached_get(memc_client_.get(), key_str.data(),
                              key_str.size(), &val_len, &cas_token, &rc);

    if (rc != MEMCACHED_SUCCESS) {
      ROMULUS_FATAL("Failed to read key for CAS: {}", key_str);
      return false;
    }

    uint64_t current_val = std::strtoull(buf, nullptr, 10);
    free(buf);

    if (current_val != expected) {
      ROMULUS_DEBUG("CAS key: {} failed (expected={}, actual={})", key_str,
                    expected, current_val);
      return false;
    }

    std::string new_val_str = std::to_string(new_value);

    rc = memcached_cas(memc_client_.get(), key_str.data(), key_str.size(),
                       new_val_str.data(), new_val_str.size(), 0, 0, cas_token);

    if (rc == MEMCACHED_SUCCESS) {
      // ROMULUS_DEBUG("CAS key: {} from {} to {}", key_str, expected, new_value);
      return true;
    } else {
      ROMULUS_FATAL("Failed to CAS key: {} from {} to {}", key_str, expected, new_value);
      return false;
    }
}


 private:
  // Returns `key` prefixed with the registry name for internal use.
  std::string MakeInternalKey(std::string_view key) {
    return std::string(name_) + "/" + std::string(key);
  }

  const std::string name_;         // Prefix for all keys in this registry
  const std::string barrier_key_;  // Memcached barrier key
  [[maybe_unused]] uint64_t expected_barrier_value_ = 0;
  [[maybe_unused]] bool wait_failed_;
  MemcachedStUniquePtr memc_client_;
};

}  // namespace romulus
