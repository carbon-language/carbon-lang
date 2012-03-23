//===-- asan_test_utils.h ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_TEST_UTILS_H
#define ASAN_TEST_UTILS_H

#if defined(_WIN32)
typedef unsigned __int8  uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
typedef __int8           int8_t;
typedef __int16          int16_t;
typedef __int32          int32_t;
typedef __int64          int64_t;
# define NOINLINE __declspec(noinline)
#else  // defined(_WIN32)
# define NOINLINE __attribute__((noinline))
#endif  // defined(_WIN32)

#if !defined(__has_feature)
#define __has_feature(x) 0
#endif

#ifndef __WORDSIZE
#if __LP64__ || defined(_WIN64)
#define __WORDSIZE 64
#else
#define __WORDSIZE 32
#endif
#endif

// Make the compiler think that something is going on there.
extern "C" void break_optimization(void *);

// This function returns its parameter but in such a way that compiler
// can not prove it.
template<class T>
NOINLINE
static T Ident(T t) {
  T ret = t;
  break_optimization(&ret);
  return ret;
}

#endif  // ASAN_TEST_UTILS_H
