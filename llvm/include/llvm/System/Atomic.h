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

#ifndef LLVM_SYSTEM_ATOMIC_H
#define LLVM_SYSTEM_ATOMIC_H

#include "llvm/Config/config.h"
#include <stdint.h>

#ifdef __APPLE__

#elif LLVM_ON_WIN32
#include <windows.h>
#endif


namespace llvm {
  namespace sys {
    
#if !defined(ENABLE_THREADS) || ENABLE_THREADS == 0
    inline void MemoryFence() {
      return;
    }
    
    typedef uint32_t cas_flag;
    inline cas_flag CompareAndSwap(cas_flag* dest, cas_flag exc, cas_flag c) {
      cas_flag result = *dest;
      if (result == c)
        *dest = exc;
      return result;
    }
    
#elif __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 1)
    inline void MemoryFence() {
      __sync_synchronize();
    }
    
    typedef volatile uint32_t cas_flag;
    inline cas_flag CompareAndSwap(cas_flag* dest, cas_flag exc, cas_flag c) {
      return __sync_val_compare_and_swap(dest, exc, c);
    }
    
#elif defined(__APPLE__)
    inline void MemoryFence() {
      OSMemoryBarrier();
    }
    
    typedef volatile UInt32 cas_flag;
    inline cas_flag CompareAndSwap(cas_flag* dest, cas_flag exc, cas_flag c) {
      cas_flag old = *dest;
      OSCompareAndSwap(c, exc, dest);
      return old;
    }
#elif defined(LLVM_ON_WIN32)
#warning Memory fence implementation requires Windows 2003 or later.
    inline void MemoryFence() {
      MemoryBarrier();
    }
    
    typedef volatile long cas_flag;
    inline cas_flag CompareAndSwap(cas_flag* dest, cas_flag exc, cas_flag c) {
      return _InterlockedCompareExchange(dest, exc, c);
    }
#else
#error No memory atomics implementation for your platform!
#endif
    
  }
}

#endif