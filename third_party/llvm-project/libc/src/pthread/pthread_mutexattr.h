//===-- Declarations related mutex attribute objects  -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEXATTR_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEXATTR_H

#include <pthread.h>

namespace __llvm_libc {

enum class PThreadMutexAttrPos : unsigned int {
  TYPE_SHIFT = 0,
  TYPE_MASK = 0x3 << TYPE_SHIFT, // Type is encoded in 2 bits

  ROBUST_SHIFT = 2,
  ROBUST_MASK = 0x1 << ROBUST_SHIFT,

  PSHARED_SHIFT = 3,
  PSHARED_MASK = 0x1 << PSHARED_SHIFT,

  // TODO: Add a mask for protocol and prioceiling when it is supported.
};

constexpr pthread_mutexattr_t DEFAULT_MUTEXATTR =
    PTHREAD_MUTEX_DEFAULT << unsigned(PThreadMutexAttrPos::TYPE_SHIFT) |
    PTHREAD_MUTEX_STALLED << unsigned(PThreadMutexAttrPos::ROBUST_SHIFT) |
    PTHREAD_PROCESS_PRIVATE << unsigned(PThreadMutexAttrPos::PSHARED_SHIFT);

static inline int get_mutexattr_type(pthread_mutexattr_t attr) {
  return (attr & unsigned(PThreadMutexAttrPos::TYPE_MASK)) >>
         unsigned(PThreadMutexAttrPos::TYPE_SHIFT);
}

static inline int get_mutexattr_robust(pthread_mutexattr_t attr) {
  return (attr & unsigned(PThreadMutexAttrPos::ROBUST_MASK)) >>
         unsigned(PThreadMutexAttrPos::ROBUST_SHIFT);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEXATTR_H
