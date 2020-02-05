//===-- runtime/lock.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Wraps pthread_mutex_t (or whatever)

#ifndef FORTRAN_RUNTIME_LOCK_H_
#define FORTRAN_RUNTIME_LOCK_H_

#include <pthread.h>

namespace Fortran::runtime {

class Lock {
public:
  Lock() { pthread_mutex_init(&mutex_, nullptr); }
  ~Lock() { pthread_mutex_destroy(&mutex_); }
  void Take() { pthread_mutex_lock(&mutex_); }
  bool Try() { return pthread_mutex_trylock(&mutex_) != 0; }
  void Drop() { pthread_mutex_unlock(&mutex_); }

  void CheckLocked(const Terminator &terminator) {
    if (Try()) {
      Drop();
      terminator.Crash("Lock::CheckLocked() failed");
    }
  }

private:
  pthread_mutex_t mutex_;
};

class CriticalSection {
public:
  explicit CriticalSection(Lock &lock) : lock_{lock} { lock_.Take(); }
  ~CriticalSection() { lock_.Drop(); }

private:
  Lock &lock_;
};
}

#endif  // FORTRAN_RUNTIME_LOCK_H_
