//===-- tsan_rtl.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Compiler-specific definitions.
//===----------------------------------------------------------------------===//

#ifndef TSAN_COMPILER_H
#define TSAN_COMPILER_H

#define INLINE        static inline
#define NOINLINE      __attribute__((noinline))
#define ALWAYS_INLINE __attribute__((always_inline))
#define NORETURN      __attribute__((noreturn))
#define WEAK          __attribute__((weak))
#define ALIGN(n)      __attribute__((aligned(n)))
#define LIKELY(x)     __builtin_expect(!!(x), 1)
#define UNLIKELY(x)   __builtin_expect(!!(x), 0)
#define THREADLOCAL   __thread
#define FORMAT(f, a)  __attribute__((format(printf, f, a)))
#define USED          __attribute__((used))

#endif  // TSAN_COMPILER_H
