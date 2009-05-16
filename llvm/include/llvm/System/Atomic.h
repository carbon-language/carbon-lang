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

#include <stdint.h>

#if defined(__APPLE__)
#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ == 0)
#include <libkern/OSAtomic.h>
#endif
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

    typedef uint32_t cas_flag;
    inline cas_flag CompareAndSwap(cas_flag* dest, cas_flag exc, cas_flag c) {
      return __sync_val_compare_and_swap(dest, exc, c);
    }

#elif defined(__APPLE__)
    inline void MemoryFence() {
      OSMemoryBarrier();
    }

    typedef int32_t cas_flag;
    inline cas_flag CompareAndSwap(cas_flag* dest, cas_flag exc, cas_flag c) {
      cas_flag old = *dest;
      OSAtomicCompareAndSwap32(c, exc, dest);
      return old;
    }
#elif defined(LLVM_ON_WIN32)
    inline void MemoryFence() {
#ifdef _MSC_VER
      MemoryBarrier();
#elif 0
      // FIXME: Requires SSE2 support
      __asm__ __volatile__("mfence":::"memory");
#else
      // FIXME: just temporary workaround. We need to emit some fence...
      __asm__ __volatile__("":::"memory");
#endif
    }

    typedef volatile long cas_flag;
    inline cas_flag CompareAndSwap(cas_flag* dest, cas_flag exc, cas_flag c) {
      return InterlockedCompareExchange(dest, exc, c);
    }
#else
#error No memory atomics implementation for your platform!
#endif

  }
}

#endif
