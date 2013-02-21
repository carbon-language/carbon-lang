//===-- asan_intercepted_functions.h ----------------------------*- C++ -*-===//
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
// ASan-private header containing prototypes for wrapper functions and wrappers
//===----------------------------------------------------------------------===//
#ifndef ASAN_INTERCEPTED_FUNCTIONS_H
#define ASAN_INTERCEPTED_FUNCTIONS_H

#include "asan_internal.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_platform_interceptors.h"

#include <stdarg.h>
#include <stddef.h>

using __sanitizer::uptr;

// Use macro to describe if specific function should be
// intercepted on a given platform.
#if !defined(_WIN32)
# define ASAN_INTERCEPT_ATOLL_AND_STRTOLL 1
# define ASAN_INTERCEPT__LONGJMP 1
# define ASAN_INTERCEPT_STRDUP 1
# define ASAN_INTERCEPT_STRCASECMP_AND_STRNCASECMP 1
# define ASAN_INTERCEPT_INDEX 1
# define ASAN_INTERCEPT_PTHREAD_CREATE 1
# define ASAN_INTERCEPT_MLOCKX 1
#else
# define ASAN_INTERCEPT_ATOLL_AND_STRTOLL 0
# define ASAN_INTERCEPT__LONGJMP 0
# define ASAN_INTERCEPT_STRDUP 0
# define ASAN_INTERCEPT_STRCASECMP_AND_STRNCASECMP 0
# define ASAN_INTERCEPT_INDEX 0
# define ASAN_INTERCEPT_PTHREAD_CREATE 0
# define ASAN_INTERCEPT_MLOCKX 0
#endif

#if defined(__linux__)
# define ASAN_USE_ALIAS_ATTRIBUTE_FOR_INDEX 1
#else
# define ASAN_USE_ALIAS_ATTRIBUTE_FOR_INDEX 0
#endif

#if !defined(__APPLE__)
# define ASAN_INTERCEPT_STRNLEN 1
#else
# define ASAN_INTERCEPT_STRNLEN 0
#endif

#if defined(__linux__) && !defined(ANDROID)
# define ASAN_INTERCEPT_SWAPCONTEXT 1
#else
# define ASAN_INTERCEPT_SWAPCONTEXT 0
#endif

#if !defined(ANDROID) && !defined(_WIN32)
# define ASAN_INTERCEPT_SIGNAL_AND_SIGACTION 1
#else
# define ASAN_INTERCEPT_SIGNAL_AND_SIGACTION 0
#endif

#if !defined(_WIN32)
# define ASAN_INTERCEPT_SIGLONGJMP 1
#else
# define ASAN_INTERCEPT_SIGLONGJMP 0
#endif

#if ASAN_HAS_EXCEPTIONS && !defined(_WIN32)
# define ASAN_INTERCEPT___CXA_THROW 1
#else
# define ASAN_INTERCEPT___CXA_THROW 0
#endif

// Windows threads.
# if defined(_WIN32)
__declspec(dllimport)
void* __stdcall CreateThread(void *sec, uptr st, void* start,
                             void *arg, DWORD fl, DWORD *id);
# endif

#endif  // ASAN_INTERCEPTED_FUNCTIONS_H
