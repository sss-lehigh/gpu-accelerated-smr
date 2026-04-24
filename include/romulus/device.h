#pragma once

#include <infiniband/verbs.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "common.h"

namespace romulus {

class Device {
 public:
  Device(uint8_t transport_flag)
      : dev_list_(ibv_get_device_list(&num_devs_)),
        transport_type_(transport_flag) {}
  ~Device() {
    pds_.clear();
    if (dev_list_ != nullptr) {
      ROMULUS_DEBUG("Closing device: {}", name_);
      ibv_free_device_list(dev_list_);
    }
  }

  bool Open() { return Open(std::nullopt, std::nullopt); }

  bool Open(std::optional<const std::string_view> name,
            std::optional<int> port) {
    if (num_devs_ < 1) {
      ROMULUS_FATAL("No devices found on this machine");
      return false;
    }
    for (int i = 0; i < num_devs_; ++i) {
      name_ = dev_list_[i]->name;
      if (name.has_value() && name.value() != name_) continue;
      ROMULUS_DEBUG("Trying to open device: name={}, port={}", name_,
                    port.value_or(-1));
      context_.reset(ibv_open_device(dev_list_[i]));
      if (!(port.has_value() ? SetPortIfActive(port.value())
                             : TryFindActivePort()) &&
          ValidateDevice()) {
        ROMULUS_DEBUG("Failed to set port for device: {}", name_);
        if (context_ != nullptr) {
          ROMULUS_ASSERT(ibv_close_device(context_.get()),
                         "Failed to close device: {}", context_->device->name);
        }
        continue;
      }
      ROMULUS_INFO("Using device: name={}, port={}", name_, port_);
      return true;
    }
    ROMULUS_FATAL("Exhausted device list while trying to open {}", name_);
    return false;
  }

  void AllocatePd(const std::string& pd_name) {
    pds_[pd_name] = PdUniquePtr(ibv_alloc_pd(context_.get()));
  }
  void DeallocPd(const std::string& pd_name) {
    pds_.erase(pd_name);  // Calls ibv_dealloc_pd
  }

  ibv_pd* GetPd(const std::string& pd_name) {
    if (pds_.find(pd_name) == pds_.end()) return nullptr;
    return pds_[pd_name].get();
  }

  int GetNumDevices() const { return num_devs_; }
  ibv_device** GetDeviceList() const { return dev_list_; }
  ibv_context* GetContext() const { return context_.get(); }
  const ibv_device_attr* GetDevAttr() const { return &dev_attr_; }
  std::string GetDeviceName() const { return name_; }
  int GetPort() const { return port_; }
  uint16_t GetLid() const { return port_attr_.lid; }

 private:
  bool TryFindActivePort() {
    ROMULUS_DEBUG("Trying to find active port for device: {}...", name_);
    ROMULUS_ASSERT(ibv_query_device(context_.get(), &dev_attr_) == 0,
                   "Failed to query device in context: {}",
                   reinterpret_cast<uintptr_t>(context_.get()));
    for (int i = 1; i < dev_attr_.phys_port_cnt + 1; ++i) {
      if (SetPortIfActive(i)) return true;
    }
    ROMULUS_DEBUG("No active port on device: {}", name_);
    return false;
  }
  bool SetPortIfActive(int port) {
    ROMULUS_DEBUG("Attempting to set port for device: {}", name_);
    ROMULUS_ASSERT(ibv_query_port(context_.get(), port, &port_attr_) == 0,
                   "Failed to query port for device: {}", name_);
    if (port_attr_.state == IBV_PORT_ACTIVE &&
        port_attr_.link_layer == transport_type_) {
      port_ = port;
      if(transport_type_ == IBV_LINK_LAYER_INFINIBAND){
        ROMULUS_INFO("Selected port {} -- INFINIBAND transport layer", port);
      }else if(transport_type_ == IBV_LINK_LAYER_ETHERNET){
        ROMULUS_INFO("Selected port {} -- RoCE transport layer", port);
      }
      return true;
    }
    return false;
  }
  bool ValidateDevice() {
    struct ibv_device_attr dev_attr;
    ROMULUS_ASSERT(ibv_query_device(context_.get(), &dev_attr) == 0,
                   "Failed to query device for validation");
    if (dev_attr.atomic_cap == IBV_ATOMIC_NONE) {
      ROMULUS_FATAL("Device does not have necessary atomic capabilities");
      return false;
    }
    return true;
  }

  int num_devs_ = -1;
  struct ibv_device** dev_list_ = nullptr;
  ContextUniquePtr context_;
  std::unordered_map<std::string, PdUniquePtr> pds_;
  std::string name_ = "";
  struct ibv_device_attr dev_attr_;
  int port_;
  struct ibv_port_attr port_attr_;
  uint8_t transport_type_;
};

}  // namespace romulus
