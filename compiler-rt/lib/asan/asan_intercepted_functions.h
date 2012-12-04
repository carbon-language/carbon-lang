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

// On Darwin siglongjmp tailcalls longjmp, so we don't want to intercept it
// there.
#if !defined(_WIN32) && (!defined(__APPLE__) || MAC_INTERPOSE_FUNCTIONS)
# define ASAN_INTERCEPT_SIGLONGJMP 1
#else
# define ASAN_INTERCEPT_SIGLONGJMP 0
#endif

#if ASAN_HAS_EXCEPTIONS && !defined(_WIN32)
# define ASAN_INTERCEPT___CXA_THROW 1
#else
# define ASAN_INTERCEPT___CXA_THROW 0
#endif

#define DECLARE_FUNCTION_AND_WRAPPER(ret_type, func, ...) \
  ret_type func(__VA_ARGS__); \
  ret_type WRAP(func)(__VA_ARGS__)

// Use extern declarations of intercepted functions on Mac and Windows
// to avoid including system headers.
#if defined(__APPLE__) || (defined(_WIN32) && !defined(_DLL))
extern "C" {
// signal.h
# if ASAN_INTERCEPT_SIGNAL_AND_SIGACTION
struct sigaction;
DECLARE_FUNCTION_AND_WRAPPER(int, sigaction, int sig,
              const struct sigaction *act,
              struct sigaction *oldact);
DECLARE_FUNCTION_AND_WRAPPER(void*, signal, int signum, void *handler);
# endif

// setjmp.h
DECLARE_FUNCTION_AND_WRAPPER(void, longjmp, void *env, int value);
# if ASAN_INTERCEPT__LONGJMP
DECLARE_FUNCTION_AND_WRAPPER(void, _longjmp, void *env, int value);
# endif
# if ASAN_INTERCEPT_SIGLONGJMP
DECLARE_FUNCTION_AND_WRAPPER(void, siglongjmp, void *env, int value);
# endif
# if ASAN_INTERCEPT___CXA_THROW
DECLARE_FUNCTION_AND_WRAPPER(void, __cxa_throw, void *a, void *b, void *c);
#endif

// string.h / strings.h
DECLARE_FUNCTION_AND_WRAPPER(int, memcmp,
                             const void *a1, const void *a2, uptr size);
DECLARE_FUNCTION_AND_WRAPPER(void*, memmove,
                             void *to, const void *from, uptr size);
DECLARE_FUNCTION_AND_WRAPPER(void*, memcpy,
                             void *to, const void *from, uptr size);
DECLARE_FUNCTION_AND_WRAPPER(void*, memset, void *block, int c, uptr size);
DECLARE_FUNCTION_AND_WRAPPER(char*, strchr, const char *str, int c);
DECLARE_FUNCTION_AND_WRAPPER(char*, strcat,  /* NOLINT */
                             char *to, const char* from);
DECLARE_FUNCTION_AND_WRAPPER(char*, strncat,
                             char *to, const char* from, uptr size);
DECLARE_FUNCTION_AND_WRAPPER(char*, strcpy,  /* NOLINT */
                             char *to, const char* from);
DECLARE_FUNCTION_AND_WRAPPER(char*, strncpy,
                             char *to, const char* from, uptr size);
DECLARE_FUNCTION_AND_WRAPPER(int, strcmp, const char *s1, const char* s2);
DECLARE_FUNCTION_AND_WRAPPER(int, strncmp,
                             const char *s1, const char* s2, uptr size);
DECLARE_FUNCTION_AND_WRAPPER(uptr, strlen, const char *s);
# if ASAN_INTERCEPT_STRCASECMP_AND_STRNCASECMP
DECLARE_FUNCTION_AND_WRAPPER(int, strcasecmp, const char *s1, const char *s2);
DECLARE_FUNCTION_AND_WRAPPER(int, strncasecmp,
                             const char *s1, const char *s2, uptr n);
# endif
# if ASAN_INTERCEPT_STRDUP
DECLARE_FUNCTION_AND_WRAPPER(char*, strdup, const char *s);
# endif
# if ASAN_INTERCEPT_STRNLEN
DECLARE_FUNCTION_AND_WRAPPER(uptr, strnlen, const char *s, uptr maxlen);
# endif
#if ASAN_INTERCEPT_INDEX
DECLARE_FUNCTION_AND_WRAPPER(char*, index, const char *string, int c);
#endif

// stdlib.h
DECLARE_FUNCTION_AND_WRAPPER(int, atoi, const char *nptr);
DECLARE_FUNCTION_AND_WRAPPER(long, atol, const char *nptr);  // NOLINT
DECLARE_FUNCTION_AND_WRAPPER(long, strtol, const char *nptr, char **endptr, int base);  // NOLINT
# if ASAN_INTERCEPT_ATOLL_AND_STRTOLL
DECLARE_FUNCTION_AND_WRAPPER(long long, atoll, const char *nptr);  // NOLINT
DECLARE_FUNCTION_AND_WRAPPER(long long, strtoll, const char *nptr, char **endptr, int base);  // NOLINT
# endif

# if ASAN_INTERCEPT_MLOCKX
// mlock/munlock
DECLARE_FUNCTION_AND_WRAPPER(int, mlock, const void *addr, size_t len);
DECLARE_FUNCTION_AND_WRAPPER(int, munlock, const void *addr, size_t len);
DECLARE_FUNCTION_AND_WRAPPER(int, mlockall, int flags);
DECLARE_FUNCTION_AND_WRAPPER(int, munlockall, void);
# endif

// Windows threads.
# if defined(_WIN32)
__declspec(dllimport)
void* __stdcall CreateThread(void *sec, uptr st, void* start,
                             void *arg, DWORD fl, DWORD *id);
# endif
// Posix threads.
# if ASAN_INTERCEPT_PTHREAD_CREATE
DECLARE_FUNCTION_AND_WRAPPER(int, pthread_create,
                             void *thread, void *attr,
                             void *(*start_routine)(void*), void *arg);
# endif

#if defined(__APPLE__)
typedef void* pthread_workqueue_t;
typedef void* pthread_workitem_handle_t;

typedef void* dispatch_group_t;
typedef void* dispatch_queue_t;
typedef void* dispatch_source_t;
typedef u64 dispatch_time_t;
typedef void (*dispatch_function_t)(void *block);
typedef void* (*worker_t)(void *block);
typedef void* CFStringRef;
typedef void* CFAllocatorRef;

DECLARE_FUNCTION_AND_WRAPPER(void, dispatch_async_f,
                             dispatch_queue_t dq,
                             void *ctxt, dispatch_function_t func);
DECLARE_FUNCTION_AND_WRAPPER(void, dispatch_sync_f,
                             dispatch_queue_t dq,
                             void *ctxt, dispatch_function_t func);
DECLARE_FUNCTION_AND_WRAPPER(void, dispatch_after_f,
                             dispatch_time_t when, dispatch_queue_t dq,
                             void *ctxt, dispatch_function_t func);
DECLARE_FUNCTION_AND_WRAPPER(void, dispatch_barrier_async_f,
                             dispatch_queue_t dq,
                             void *ctxt, dispatch_function_t func);
DECLARE_FUNCTION_AND_WRAPPER(void, dispatch_group_async_f,
                             dispatch_group_t group, dispatch_queue_t dq,
                             void *ctxt, dispatch_function_t func);

DECLARE_FUNCTION_AND_WRAPPER(void, __CFInitialize, void);
DECLARE_FUNCTION_AND_WRAPPER(CFStringRef, CFStringCreateCopy,
                             CFAllocatorRef alloc, CFStringRef str);
DECLARE_FUNCTION_AND_WRAPPER(void, free, void* ptr);
#if MAC_INTERPOSE_FUNCTIONS && !defined(MISSING_BLOCKS_SUPPORT)
DECLARE_FUNCTION_AND_WRAPPER(void, dispatch_group_async,
                             dispatch_group_t dg,
                             dispatch_queue_t dq, void (^work)(void));
DECLARE_FUNCTION_AND_WRAPPER(void, dispatch_async,
                             dispatch_queue_t dq, void (^work)(void));
DECLARE_FUNCTION_AND_WRAPPER(void, dispatch_after,
                             dispatch_queue_t dq, void (^work)(void));
DECLARE_FUNCTION_AND_WRAPPER(void, dispatch_source_set_event_handler,
                             dispatch_source_t ds, void (^work)(void));
DECLARE_FUNCTION_AND_WRAPPER(void, dispatch_source_set_cancel_handler,
                             dispatch_source_t ds, void (^work)(void));
#endif  // MAC_INTERPOSE_FUNCTIONS
#endif  // __APPLE__
}  // extern "C"
#endif

#endif  // ASAN_INTERCEPTED_FUNCTIONS_H
