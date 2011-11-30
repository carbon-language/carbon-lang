//===-- asan_lock.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// A wrapper for a simple lock.
//===----------------------------------------------------------------------===//
#ifndef ASAN_LOCK_H
#define ASAN_LOCK_H

#include "asan_internal.h"

// The locks in ASan are global objects and they are never destroyed to avoid
// at-exit races (that is, a lock is being used by other threads while the main
// thread is doing atexit destructors).

#ifdef __APPLE__
#include <pthread.h>

#include <libkern/OSAtomic.h>
namespace __asan {
class AsanLock {
 public:
  explicit AsanLock(LinkerInitialized) :
    mu_(OS_SPINLOCK_INIT),
    owner_(0),
    is_locked_(false) {}

  void Lock() {
    CHECK(owner_ != pthread_self());
    OSSpinLockLock(&mu_);
    is_locked_ = true;
    owner_ = pthread_self();
  }
  void Unlock() {
    owner_ = 0;
    is_locked_ = false;
    OSSpinLockUnlock(&mu_);
  }

  bool IsLocked() {
    // This is not atomic, e.g. one thread may get different values if another
    // one is about to release the lock.
    return is_locked_;
  }
 private:
  OSSpinLock mu_;
  volatile pthread_t owner_;  // for debugging purposes
  bool is_locked_;  // for silly malloc_introspection_t interface
};
}  // namespace __asan

#else  // assume linux
#include <pthread.h>
namespace __asan {
class AsanLock {
 public:
  explicit AsanLock(LinkerInitialized) {
    // We assume that pthread_mutex_t initialized to all zeroes is a valid
    // unlocked mutex. We can not use PTHREAD_MUTEX_INITIALIZER as it triggers
    // a gcc warning:
    // extended initializer lists only available with -std=c++0x or -std=gnu++0x
  }
  void Lock() {
    pthread_mutex_lock(&mu_);
    // pthread_spin_lock(&mu_);
  }
  void Unlock() {
    pthread_mutex_unlock(&mu_);
    // pthread_spin_unlock(&mu_);
  }
 private:
  pthread_mutex_t mu_;
  // pthread_spinlock_t mu_;
};
}  // namespace __asan
#endif

namespace __asan {
class ScopedLock {
 public:
  explicit ScopedLock(AsanLock *mu) : mu_(mu) {
    mu_->Lock();
  }
  ~ScopedLock() {
    mu_->Unlock();
  }
 private:
  AsanLock *mu_;
};

}  // namespace __asan

#endif  // ASAN_LOCK_H
