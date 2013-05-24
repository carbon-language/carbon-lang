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
#if !SANITIZER_WINDOWS
# define ASAN_INTERCEPT_ATOLL_AND_STRTOLL 1
# define ASAN_INTERCEPT__LONGJMP 1
# define ASAN_INTERCEPT_STRDUP 1
# define ASAN_INTERCEPT_INDEX 1
# define ASAN_INTERCEPT_PTHREAD_CREATE 1
# define ASAN_INTERCEPT_MLOCKX 1
#else
# define ASAN_INTERCEPT_ATOLL_AND_STRTOLL 0
# define ASAN_INTERCEPT__LONGJMP 0
# define ASAN_INTERCEPT_STRDUP 0
# define ASAN_INTERCEPT_INDEX 0
# define ASAN_INTERCEPT_PTHREAD_CREATE 0
# define ASAN_INTERCEPT_MLOCKX 0
#endif

#if SANITIZER_LINUX
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

#if !SANITIZER_ANDROID && !SANITIZER_WINDOWS
# define ASAN_INTERCEPT_SIGNAL_AND_SIGACTION 1
#else
# define ASAN_INTERCEPT_SIGNAL_AND_SIGACTION 0
#endif

#if !SANITIZER_WINDOWS
# define ASAN_INTERCEPT_SIGLONGJMP 1
#else
# define ASAN_INTERCEPT_SIGLONGJMP 0
#endif

#if ASAN_HAS_EXCEPTIONS && !SANITIZER_WINDOWS
# define ASAN_INTERCEPT___CXA_THROW 1
#else
# define ASAN_INTERCEPT___CXA_THROW 0
#endif

#if !SANITIZER_WINDOWS
# define ASAN_INTERCEPT___CXA_ATEXIT 1
#else
# define ASAN_INTERCEPT___CXA_ATEXIT 0
#endif

# if SANITIZER_WINDOWS
extern "C" {
// Windows threads.
__declspec(dllimport)
void* __stdcall CreateThread(void *sec, uptr st, void* start,
                             void *arg, DWORD fl, DWORD *id);

int memcmp(const void *a1, const void *a2, uptr size);
void memmove(void *to, const void *from, uptr size);
void* memset(void *block, int c, uptr size);
void* memcpy(void *to, const void *from, uptr size);
char* strcat(char *to, const char* from);  // NOLINT
char* strchr(const char *str, int c);
int strcmp(const char *s1, const char* s2);
char* strcpy(char *to, const char* from);  // NOLINT
uptr strlen(const char *s);
char* strncat(char *to, const char* from, uptr size);
int strncmp(const char *s1, const char* s2, uptr size);
char* strncpy(char *to, const char* from, uptr size);
uptr strnlen(const char *s, uptr maxlen);
int atoi(const char *nptr);
long atol(const char *nptr);  // NOLINT
long strtol(const char *nptr, char **endptr, int base);  // NOLINT
void longjmp(void *env, int value);
double frexp(double x, int *expptr);
}
# endif

#endif  // ASAN_INTERCEPTED_FUNCTIONS_H
