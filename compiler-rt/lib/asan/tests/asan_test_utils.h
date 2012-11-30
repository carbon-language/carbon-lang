//===-- asan_test_utils.h ---------------------------------------*- C++ -*-===//
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

#if !defined(ASAN_EXTERNAL_TEST_CONFIG)
# define INCLUDED_FROM_ASAN_TEST_UTILS_H
# include "asan_test_config.h"
# undef INCLUDED_FROM_ASAN_TEST_UTILS_H
#endif

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
# define USED
#else  // defined(_WIN32)
# define NOINLINE __attribute__((noinline))
# define USED __attribute__((used))
#endif  // defined(_WIN32)

#if !defined(__has_feature)
#define __has_feature(x) 0
#endif

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
# define ATTRIBUTE_NO_ADDRESS_SAFETY_ANALYSIS \
    __attribute__((no_address_safety_analysis))
#else
# define ATTRIBUTE_NO_ADDRESS_SAFETY_ANALYSIS
#endif

#if __LP64__ || defined(_WIN64)
#  define SANITIZER_WORDSIZE 64
#else
#  define SANITIZER_WORDSIZE 32
#endif

// Make the compiler thinks that something is going on there.
inline void break_optimization(void *arg) {
  __asm__ __volatile__ ("" : : "r" (arg) : "memory");
}

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
