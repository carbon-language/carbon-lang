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

#include "llvm/Support/DataTypes.h"

namespace llvm {
  namespace sys {
    void MemoryFence();

    uint32_t CompareAndSwap32(volatile uint32_t* ptr,
                            uint32_t new_value,
                            uint32_t old_value);
    int32_t AtomicIncrement32(volatile int32_t* ptr);
    int32_t AtomicDecrement32(volatile int32_t* ptr);
    int32_t AtomicAdd32(volatile int32_t* ptr, int32_t val);
    
    int64_t AtomicAdd64(volatile int64_t* ptr, int64_t val);
  }
}

#endif
