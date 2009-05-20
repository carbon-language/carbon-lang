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

#if defined(_MSC_VER)
#define NOMINMAX
#include <windows.h>
#endif


namespace llvm {
  namespace sys {
    
    inline void MemoryFence() {
#if LLVM_MULTITHREADED==0
      return;
#else
#  if defined(__GNUC__)
      __sync_synchronize();
#  elif defined(_MSC_VER)
      MemoryBarrier();
#  else
#    error No memory fence implementation for your platform!
#  endif
#endif
}

#if LLVM_MULTITHREADED==0
    typedef unsigned long cas_flag;
    template<typename T>
    inline T CompareAndSwap(volatile T* dest,
			    T exc, T c) {
      T result = *dest;
      if (result == c)
        *dest = exc;
      return result;
    }
#elif defined(__GNUC__)
    typedef unsigned long cas_flag;
    template<typename T>
    inline T CompareAndSwap(volatile T* ptr,
			    T new_value,
			    T old_value) {
      return __sync_val_compare_and_swap(ptr, old_value, new_value);
    }
#elif defined(_MSC_VER)
    typedef LONG cas_flag;
    template<typename T>
    inline T CompareAndSwap(volatile T* ptr,
			    T new_value,
			    T old_value) {
      if (sizeof(T) == 4)
	return InterlockedCompareExchange(ptr, new_value, old_value);
      else if (sizeof(T) == 8)
	return InterlockedCompareExchange64(ptr, new_value, old_value);
      else
	assert(0 && "Unsupported compare-and-swap size!");
    }
    
    template<typename T>
    inline T* CompareAndSwap<T*>(volatile T** ptr,
				 T* new_value,
				 T* old_value) {
      return InterlockedCompareExchangePtr(ptr, new_value, old_value);
    }


#else
#  error No compare-and-swap implementation for your platform!
#endif

  }
}

#endif
