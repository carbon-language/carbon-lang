//===-- sanitizer_mutex.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_MUTEX_H
#define SANITIZER_MUTEX_H

#include "sanitizer_internal_defs.h"
#include "sanitizer_atomic.h"

namespace __sanitizer {

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

}  // namespace __sanitizer

#endif  // SANITIZER_MUTEX_H
