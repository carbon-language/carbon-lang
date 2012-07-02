//===-- sanitizer_mutex.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_MUTEX_H
#define SANITIZER_MUTEX_H

#include "sanitizer_atomic.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"

namespace __sanitizer {

class SpinMutex {
 public:
  SpinMutex() {
    atomic_store(&state_, 0, memory_order_relaxed);
  }

  void Lock() {
    if (atomic_exchange(&state_, 1, memory_order_acquire) == 0)
      return;
    LockSlow();
  }

  void Unlock() {
    atomic_store(&state_, 0, memory_order_release);
  }

 private:
  atomic_uint8_t state_;

  void NOINLINE LockSlow() {
    for (int i = 0;; i++) {
      if (i < 10)
        proc_yield(10);
      else
        internal_sched_yield();
      if (atomic_load(&state_, memory_order_relaxed) == 0
          && atomic_exchange(&state_, 1, memory_order_acquire) == 0)
        return;
    }
  }

  SpinMutex(const SpinMutex&);
  void operator=(const SpinMutex&);
};

template<typename MutexType>
class GenericScopedLock {
 public:
  explicit GenericScopedLock(MutexType *mu)
      : mu_(mu) {
    mu_->Lock();
  }

  ~GenericScopedLock() {
    mu_->Unlock();
  }

 private:
  MutexType *mu_;

  GenericScopedLock(const GenericScopedLock&);
  void operator=(const GenericScopedLock&);
};

template<typename MutexType>
class GenericScopedReadLock {
 public:
  explicit GenericScopedReadLock(MutexType *mu)
      : mu_(mu) {
    mu_->ReadLock();
  }

  ~GenericScopedReadLock() {
    mu_->ReadUnlock();
  }

 private:
  MutexType *mu_;

  GenericScopedReadLock(const GenericScopedReadLock&);
  void operator=(const GenericScopedReadLock&);
};

typedef GenericScopedLock<SpinMutex> SpinMutexLock;

}  // namespace __sanitizer

#endif  // SANITIZER_MUTEX_H
