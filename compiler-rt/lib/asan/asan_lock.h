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
// We define the class using opaque storage to avoid including system headers.

namespace __asan {

class AsanLock {
 public:
  explicit AsanLock(LinkerInitialized);
  void Lock();
  void Unlock();
  bool IsLocked() { return owner_ != 0; }
 private:
  uptr opaque_storage_[10];
  uptr owner_;  // for debugging and for malloc_introspection_t interface
};

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
