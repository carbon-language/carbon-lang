//===-- Support/ThreadSupport.h - Generic threading support -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines platform-agnostic interfaces that can be used to write
// multi-threaded programs.  Autoconf is used to chose the correct
// implementation of these interfaces, or default to a non-thread-capable system
// if no matching system support is available.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_THREADSUPPORT_H
#define SUPPORT_THREADSUPPORT_H

#if @HAVE_PTHREAD_MUTEX_LOCK@
#include "Support/ThreadSupport-PThreads.h"
#else
#include "Support/ThreadSupport-NoSupport.h"
#endif // If no system support is available

namespace llvm {
  /// MutexLocker - Instances of this class acquire a given Lock when
  /// constructed and hold that lock until destruction.
  ///
  class MutexLocker {
    Mutex &M;
    MutexLocker(const MutexLocker &);    // DO NOT IMPLEMENT
    void operator=(const MutexLocker &); // DO NOT IMPLEMENT
  public:
    MutexLocker(Mutex &m) : M(m) { M.acquire(); }
    ~MutexLocker() { M.release(); }
  };
}

#endif // SUPPORT_THREADSUPPORT_H
