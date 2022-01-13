//===--- amdgpu/impl/impl.cpp ------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "rt.h"
#include <memory>

/*
 * Data
 */

// host pointer (either src or dest) must be locked via hsa_amd_memory_lock
static hsa_status_t invoke_hsa_copy(hsa_signal_t signal, void *dest,
                                    hsa_agent_t agent, const void *src,
                                    size_t size) {
  const hsa_signal_value_t init = 1;
  const hsa_signal_value_t success = 0;
  hsa_signal_store_screlease(signal, init);

  hsa_status_t err = hsa_amd_memory_async_copy(dest, agent, src, agent, size, 0,
                                               nullptr, signal);
  if (err != HSA_STATUS_SUCCESS)
    return err;

  // async_copy reports success by decrementing and failure by setting to < 0
  hsa_signal_value_t got = init;
  while (got == init)
    got = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_NE, init,
                                    UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  if (got != success)
    return HSA_STATUS_ERROR;

  return err;
}

struct implFreePtrDeletor {
  void operator()(void *p) {
    core::Runtime::Memfree(p); // ignore failure to free
  }
};

enum CopyDirection { H2D, D2H };

static hsa_status_t locking_async_memcpy(enum CopyDirection direction,
                                         hsa_signal_t signal, void *dest,
                                         hsa_agent_t agent, void *src,
                                         void *lockingPtr, size_t size) {
  hsa_status_t err;

  void *lockedPtr = nullptr;
  err = hsa_amd_memory_lock(lockingPtr, size, nullptr, 0, (void **)&lockedPtr);
  if (err != HSA_STATUS_SUCCESS)
    return err;

  switch (direction) {
  case H2D:
    err = invoke_hsa_copy(signal, dest, agent, lockedPtr, size);
    break;
  case D2H:
    err = invoke_hsa_copy(signal, lockedPtr, agent, src, size);
    break;
  default:
    err = HSA_STATUS_ERROR; // fall into unlock before returning
  }

  if (err != HSA_STATUS_SUCCESS) {
    // do not leak locked host pointers, but discard potential error message
    hsa_amd_memory_unlock(lockingPtr);
    return err;
  }

  err = hsa_amd_memory_unlock(lockingPtr);
  if (err != HSA_STATUS_SUCCESS)
    return err;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t impl_memcpy_h2d(hsa_signal_t signal, void *deviceDest,
                             void *hostSrc, size_t size,
                             hsa_agent_t device_agent,
                             hsa_amd_memory_pool_t MemoryPool) {
  hsa_status_t err;

  err = locking_async_memcpy(CopyDirection::H2D, signal, deviceDest,
                             device_agent, hostSrc, hostSrc, size);

  if (err == HSA_STATUS_SUCCESS)
    return err;

  // async memcpy sometimes fails in situations where
  // allocate + copy succeeds. Looks like it might be related to
  // locking part of a read only segment. Fall back for now.
  void *tempHostPtr;
  hsa_status_t ret = core::Runtime::HostMalloc(&tempHostPtr, size, MemoryPool);
  if (ret != HSA_STATUS_SUCCESS) {
    DP("HostMalloc: Unable to alloc %zu bytes for temp scratch\n", size);
    return ret;
  }
  std::unique_ptr<void, implFreePtrDeletor> del(tempHostPtr);
  memcpy(tempHostPtr, hostSrc, size);

  return locking_async_memcpy(CopyDirection::H2D, signal, deviceDest,
                              device_agent, tempHostPtr, tempHostPtr, size);
}

hsa_status_t impl_memcpy_d2h(hsa_signal_t signal, void *hostDest,
                             void *deviceSrc, size_t size,
                             hsa_agent_t deviceAgent,
                             hsa_amd_memory_pool_t MemoryPool) {
  hsa_status_t err;

  // device has always visibility over both pointers, so use that
  err = locking_async_memcpy(CopyDirection::D2H, signal, hostDest, deviceAgent,
                             deviceSrc, hostDest, size);

  if (err == HSA_STATUS_SUCCESS)
    return err;

  // hsa_memory_copy sometimes fails in situations where
  // allocate + copy succeeds. Looks like it might be related to
  // locking part of a read only segment. Fall back for now.
  void *tempHostPtr;
  hsa_status_t ret = core::Runtime::HostMalloc(&tempHostPtr, size, MemoryPool);
  if (ret != HSA_STATUS_SUCCESS) {
    DP("HostMalloc: Unable to alloc %zu bytes for temp scratch\n", size);
    return ret;
  }
  std::unique_ptr<void, implFreePtrDeletor> del(tempHostPtr);

  err = locking_async_memcpy(CopyDirection::D2H, signal, tempHostPtr,
                             deviceAgent, deviceSrc, tempHostPtr, size);
  if (err != HSA_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  memcpy(hostDest, tempHostPtr, size);
  return HSA_STATUS_SUCCESS;
}
