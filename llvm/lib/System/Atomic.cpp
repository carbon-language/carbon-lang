//===-- Atomic.cpp - Atomic Operations --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file implements atomic operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/System/Atomic.h"
#include "llvm/Config/config.h"

using namespace llvm;

#if defined(_MSC_VER)
#include <windows.h>
#undef MemoryFence
#endif

void sys::MemoryFence() {
#if LLVM_MULTITHREADED==0
  return;
#else
#  if defined(__GNUC__)
  __sync_synchronize();
#  elif defined(_MSC_VER)
  MemoryBarrier();
#  else
# error No memory fence implementation for your platform!
#  endif
#endif
}

sys::cas_flag sys::CompareAndSwap(volatile sys::cas_flag* ptr,
                                  sys::cas_flag new_value,
                                  sys::cas_flag old_value) {
#if LLVM_MULTITHREADED==0
  sys::cas_flag result = *ptr;
  if (result == old_value)
    *ptr = new_value;
  return result;
#elif defined(__GNUC__)
  return __sync_val_compare_and_swap(ptr, old_value, new_value);
#elif defined(_MSC_VER)
  return InterlockedCompareExchange(ptr, new_value, old_value);
#else
#  error No compare-and-swap implementation for your platform!
#endif
}

sys::cas_flag sys::AtomicIncrement(volatile sys::cas_flag* ptr) {
#if LLVM_MULTITHREADED==0
  ++(*ptr);
  return *ptr;
#elif defined(__GNUC__)
  return __sync_add_and_fetch(ptr, 1);
#elif defined(_MSC_VER)
  return InterlockedIncrement(ptr);
#else
#  error No atomic increment implementation for your platform!
#endif
}

sys::cas_flag sys::AtomicDecrement(volatile sys::cas_flag* ptr) {
#if LLVM_MULTITHREADED==0
  --(*ptr);
  return *ptr;
#elif defined(__GNUC__)
  return __sync_sub_and_fetch(ptr, 1);
#elif defined(_MSC_VER)
  return InterlockedDecrement(ptr);
#else
#  error No atomic decrement implementation for your platform!
#endif
}

sys::cas_flag sys::AtomicAdd(volatile sys::cas_flag* ptr, sys::cas_flag val) {
#if LLVM_MULTITHREADED==0
  *ptr += val;
  return *ptr;
#elif defined(__GNUC__)
  return __sync_add_and_fetch(ptr, val);
#elif defined(_MSC_VER)
  return InterlockedExchangeAdd(ptr, val) + val;
#else
#  error No atomic add implementation for your platform!
#endif
}

sys::cas_flag sys::AtomicMul(volatile sys::cas_flag* ptr, sys::cas_flag val) {
  sys::cas_flag original, result;
  do {
    original = *ptr;
    result = original * val;
  } while (sys::CompareAndSwap(ptr, result, original) != original);

  return result;
}

sys::cas_flag sys::AtomicDiv(volatile sys::cas_flag* ptr, sys::cas_flag val) {
  sys::cas_flag original, result;
  do {
    original = *ptr;
    result = original / val;
  } while (sys::CompareAndSwap(ptr, result, original) != original);

  return result;
}
