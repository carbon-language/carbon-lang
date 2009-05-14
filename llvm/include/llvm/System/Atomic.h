//===- llvm/System/Atomic.h - Atomic Operations -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys atomic operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"

#ifdef __APPLE__
#include <libkern/OSAtomic.h>
#elif LLVM_ON_WIN32
#include <windows.h>
#endif


#ifndef LLVM_SYSTEM_ATOMIC_H
#define LLVM_SYSTEM_ATOMIC_H

namespace llvm {
  namespace sys {
    inline void MemoryFence() {
#if !defined(ENABLE_THREADS) || ENABLE_THREADS == 0
      return;
#elif __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 1)
      __sync_synchronize();
#elif defined(__APPLE__)
      OSMemoryBarrier();
#elif defined(LLVM_ON_WIN32)
#warning Memory fence implementation requires Windows 2003 or later.
      MemoryBarrier();
#else
#warning No memory fence implementation found for you platform!
#endif
    }
  }
}

#endif