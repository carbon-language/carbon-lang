//===-- asan_interceptors.h -------------------------------------*- C++ -*-===//
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
// ASan-private header for asan_interceptors.cc
//===----------------------------------------------------------------------===//
#ifndef ASAN_INTERCEPTORS_H
#define ASAN_INTERCEPTORS_H

#include "asan_internal.h"
#include "sanitizer_common/sanitizer_interception.h"
#include "sanitizer_common/sanitizer_platform_interceptors.h"

// Use macro to describe if specific function should be
// intercepted on a given platform.
#if !SANITIZER_WINDOWS
# define ASAN_INTERCEPT_ATOLL_AND_STRTOLL 1
# define ASAN_INTERCEPT__LONGJMP 1
# define ASAN_INTERCEPT_STRDUP 1
# define ASAN_INTERCEPT_INDEX 1
# define ASAN_INTERCEPT_PTHREAD_CREATE 1
# define ASAN_INTERCEPT_FORK 1
#else
# define ASAN_INTERCEPT_ATOLL_AND_STRTOLL 0
# define ASAN_INTERCEPT__LONGJMP 0
# define ASAN_INTERCEPT_STRDUP 0
# define ASAN_INTERCEPT_INDEX 0
# define ASAN_INTERCEPT_PTHREAD_CREATE 0
# define ASAN_INTERCEPT_FORK 0
#endif

#if SANITIZER_FREEBSD || SANITIZER_LINUX
# define ASAN_USE_ALIAS_ATTRIBUTE_FOR_INDEX 1
#else
# define ASAN_USE_ALIAS_ATTRIBUTE_FOR_INDEX 0
#endif

#if !SANITIZER_MAC
# define ASAN_INTERCEPT_STRNLEN 1
#else
# define ASAN_INTERCEPT_STRNLEN 0
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
# define ASAN_INTERCEPT_SWAPCONTEXT 1
#else
# define ASAN_INTERCEPT_SWAPCONTEXT 0
#endif

#if !SANITIZER_WINDOWS
# define ASAN_INTERCEPT_SIGNAL_AND_SIGACTION 1
#else
# define ASAN_INTERCEPT_SIGNAL_AND_SIGACTION 0
#endif

#if !SANITIZER_WINDOWS
# define ASAN_INTERCEPT_SIGLONGJMP 1
#else
# define ASAN_INTERCEPT_SIGLONGJMP 0
#endif

// Android bug: https://code.google.com/p/android/issues/detail?id=61799
#if ASAN_HAS_EXCEPTIONS && !SANITIZER_WINDOWS && \
    !(SANITIZER_ANDROID && defined(__i386))
# define ASAN_INTERCEPT___CXA_THROW 1
#else
# define ASAN_INTERCEPT___CXA_THROW 0
#endif

#if !SANITIZER_WINDOWS
# define ASAN_INTERCEPT___CXA_ATEXIT 1
#else
# define ASAN_INTERCEPT___CXA_ATEXIT 0
#endif

DECLARE_REAL(int, memcmp, const void *a1, const void *a2, uptr size)
DECLARE_REAL(void*, memcpy, void *to, const void *from, uptr size)
DECLARE_REAL(void*, memset, void *block, int c, uptr size)
DECLARE_REAL(char*, strchr, const char *str, int c)
DECLARE_REAL(SIZE_T, strlen, const char *s)
DECLARE_REAL(char*, strncpy, char *to, const char *from, uptr size)
DECLARE_REAL(uptr, strnlen, const char *s, uptr maxlen)
DECLARE_REAL(char*, strstr, const char *s1, const char *s2)
struct sigaction;
DECLARE_REAL(int, sigaction, int signum, const struct sigaction *act,
                             struct sigaction *oldact)

namespace __asan {

void InitializeAsanInterceptors();

#define ENSURE_ASAN_INITED() do { \
  CHECK(!asan_init_is_running); \
  if (UNLIKELY(!asan_inited)) { \
    AsanInitFromRtl(); \
  } \
} while (0)

}  // namespace __asan

#endif  // ASAN_INTERCEPTORS_H
