//===--- amdgpu/impl/rt.h ----------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SRC_RUNTIME_INCLUDE_RT_H_
#define SRC_RUNTIME_INCLUDE_RT_H_

#include "impl_runtime.h"
#include "hsa_api.h"
#include <string>

namespace core {

#define DEFAULT_MAX_QUEUE_SIZE 4096
#define DEFAULT_DEBUG_MODE 0
class Environment {
public:
  Environment()
      : max_queue_size_(DEFAULT_MAX_QUEUE_SIZE),
        debug_mode_(DEFAULT_DEBUG_MODE) {
    GetEnvAll();
  }

  void GetEnvAll();

  int getMaxQueueSize() const { return max_queue_size_; }
  int getDebugMode() const { return debug_mode_; }

private:
  std::string GetEnv(const char *name) {
    char *env = getenv(name);
    std::string ret;
    if (env) {
      ret = env;
    }
    return ret;
  }

  int max_queue_size_;
  int debug_mode_;
};

class Runtime final {
public:
  static Runtime &getInstance() {
    static Runtime instance;
    return instance;
  }

  // modules
  static hsa_status_t RegisterModuleFromMemory(
      void *, size_t, hsa_agent_t agent,
      hsa_status_t (*on_deserialized_data)(void *data, size_t size,
                                           void *cb_state),
      void *cb_state, std::vector<hsa_executable_t> &HSAExecutables);

  // data
  static hsa_status_t Memcpy(hsa_signal_t, void *, const void *, size_t);
  static hsa_status_t Memfree(void *);
  static hsa_status_t HostMalloc(void **ptr, size_t size,
                                 hsa_amd_memory_pool_t MemoryPool);

  int getMaxQueueSize() const { return env_.getMaxQueueSize(); }
  int getDebugMode() const { return env_.getDebugMode(); }

protected:
  Runtime() = default;
  ~Runtime() = default;
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

protected:
  // variable to track environment variables
  Environment env_;
};

} // namespace core

#endif // SRC_RUNTIME_INCLUDE_RT_H_
